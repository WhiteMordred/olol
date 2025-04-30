"""Server for RPC-based distributed LLM inference."""

import json
import hashlib
import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import grpc
import grpc.aio
import requests

try:
    import numpy as np
except ImportError:
    np = None  # Handle case where numpy is not available

# Import proto definitions safely with fallback
try:
    from ..proto import ollama_pb2, ollama_pb2_grpc
except ImportError:
    try:
        import ollama_pb2
        import ollama_pb2_grpc
    except ImportError:
        # Will be generated at runtime
        pass

logger = logging.getLogger(__name__)


class RPCServer(ollama_pb2_grpc.DistributedOllamaServiceServicer):
    """Server for distributed LLM inference using RPC.
    
    This server can process subsets of model layers assigned by
    a coordinator, similar to the llama.cpp RPC architecture.
    """
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 device_type: str = "cpu",
                 device_id: int = 0) -> None:
        """Initialize the RPC server.
        
        Args:
            ollama_host: URL of the Ollama HTTP API
            device_type: Type of compute device ("cpu", "cuda", "metal")
            device_id: Device ID for multi-device systems
        """
        self.ollama_host = ollama_host
        self.device_type = device_type
        self.device_id = device_id
        self.loaded_models = set()
        self.active_computations = {}
        self.start_time = time.time()
        
        # Get device capabilities
        self.device_capabilities = self._detect_device_capabilities()
        logger.info(f"Server initialized with device: {device_type}:{device_id}")
        logger.info(f"Device capabilities: {self.device_capabilities}")
        
    def _detect_device_capabilities(self) -> Dict[str, Any]:
        """Detect device capabilities for this machine.
        
        Returns:
            Dictionary with device capability information
        """
        capabilities = {}
        system_info = self._get_system_info()
        
        # Set basic device type from system info
        if system_info.get("has_cuda", False):
            self.device_type = "cuda"
        elif system_info.get("has_rocm", False):
            self.device_type = "rocm"
        elif system_info.get("has_mps", False):
            self.device_type = "mps"
        else:
            self.device_type = "cpu"
            
        capabilities["backend_type"] = self.device_type
        
        # Add all CPU information
        if "cpu" in system_info:
            cpu_info = system_info["cpu"]
            capabilities.update({
                "cores": cpu_info.get("cores", 0),
                "threads": cpu_info.get("threads", 0),
                "architecture": cpu_info.get("architecture", "unknown"),
                "cpu_model": cpu_info.get("model", "unknown"),
                "cpu_frequency_mhz": cpu_info.get("frequency_mhz", 0),
                "cpu_max_frequency_mhz": cpu_info.get("max_frequency_mhz", 0),
                "cpu_utilization": cpu_info.get("cpu_utilization", 0),
                "cpu_vendor": cpu_info.get("vendor", "unknown"),
                "cpu_physical_cores": cpu_info.get("physical_cores", 0)
            })
        
        # Add all memory information
        if "memory" in system_info:
            memory_info = system_info["memory"]
            capabilities.update({
                "memory": memory_info.get("total_bytes", 0),  # Keeping the original field
                "memory_free_bytes": memory_info.get("free_bytes", 0),
                "memory_used_bytes": memory_info.get("used_bytes", 0),
                "memory_percent_used": memory_info.get("percent_used", 0),
                "swap_total_bytes": memory_info.get("swap_total", 0),
                "swap_free_bytes": memory_info.get("swap_free", 0),
                "swap_percent_used": memory_info.get("swap_percent", 0)
            })
        
        # Add all GPU information if available
        if "gpu" in system_info:
            gpu_info = system_info["gpu"]
            
            # Standard GPU fields
            gpu_fields = {
                "gpu_name": gpu_info.get("name", "unknown"),
                "gpu_vendor": gpu_info.get("vendor", "unknown"),
                "gpu_driver_version": gpu_info.get("driver_version", "unknown"),
                "gpu_memory_total_bytes": gpu_info.get("memory_total", 0),
                "gpu_memory_free_bytes": gpu_info.get("memory_free", 0),
                "gpu_memory_used_bytes": gpu_info.get("memory_used", 0),
                "gpu_utilization_percent": gpu_info.get("utilization_percent", 0),
                "gpu_temperature_c": gpu_info.get("temperature_c", 0),
                "gpu_power_usage_watts": gpu_info.get("power_usage_watts", 0),
                "gpu_power_limit_watts": gpu_info.get("power_limit_watts", 0),
                "gpu_compute_capability": gpu_info.get("compute_capability", ""),
                "gpu_count": gpu_info.get("count", 1),
                "gpu_clock_mhz": gpu_info.get("clock_mhz", 0),
                "gpu_memory_clock_mhz": gpu_info.get("memory_clock_mhz", 0)
            }
            
            # Add estimated TFLOPS if possible
            if "tflops" in gpu_info:
                gpu_fields["gpu_tflops"] = gpu_info["tflops"]
            elif "clock_mhz" in gpu_info and "cuda_cores" in gpu_info:
                # Estimate TFLOPS for NVIDIA using clock speed and CUDA cores
                # Formula: (clock_MHz * 2 * cuda_cores) / 1_000_000
                clock = gpu_info["clock_mhz"]
                cores = gpu_info["cuda_cores"]
                estimated_tflops = (clock * 2 * cores) / 1_000_000
                gpu_fields["gpu_tflops_estimated"] = estimated_tflops
            
            capabilities.update(gpu_fields)
            
            # Set compute capability and device ID
            if "compute_capability" in gpu_info:
                capabilities["compute_capability"] = gpu_info["compute_capability"]
            
            # Device ID can be the PCI ID or another unique identifier
            if "device_id" in gpu_info:
                capabilities["device_id"] = gpu_info["device_id"]
            elif "uuid" in gpu_info:
                capabilities["device_id"] = gpu_info["uuid"]
            else:
                # Generate a unique ID based on system info
                unique_id = hashlib.md5(
                    f"{platform.node()}-{capabilities.get('gpu_name', '')}-{os.getpid()}".encode()
                ).hexdigest()
                capabilities["device_id"] = unique_id
        else:
            # For CPU-only systems, use the CPU info for device ID
            unique_id = hashlib.md5(
                f"{platform.node()}-{capabilities.get('cpu_model', '')}-{os.getpid()}".encode()
            ).hexdigest()
            capabilities["device_id"] = unique_id
        
        # Add disk information
        capabilities.update(self._get_disk_info())
        
        # Add system information
        capabilities.update({
            "hostname": platform.node(),
            "os": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "process_id": os.getpid()
        })
        
        # Get Ollama specific capabilities
        ollama_capabilities = self._get_ollama_capabilities()
        capabilities.update(ollama_capabilities)
        
        return capabilities

    def _get_system_info(self) -> Dict[str, Any]:
        """Collecter des informations détaillées sur le système.
        
        Returns:
            Dictionary avec les informations système (CPU, RAM, GPU)
        """
        import platform
        import psutil
        
        try:
            import py3nvml.py3nvml as nvml
            has_nvml = True
        except ImportError:
            has_nvml = False
            
        try:
            import GPUtil
            has_gputil = True
        except ImportError:
            has_gputil = False
            
        system_info = {}
        
        # CPU Information
        cpu_info = {
            "cores": psutil.cpu_count(logical=False) or 0,
            "threads": psutil.cpu_count(logical=True) or 0,
            "architecture": platform.machine(),
            "cpu_utilization": psutil.cpu_percent(interval=0.1),
            "vendor": "unknown",
            "model": "unknown"
        }
        
        # Essayer d'obtenir des informations CPU plus détaillées selon le système d'exploitation
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    
                for line in cpuinfo.split("\n"):
                    if "model name" in line:
                        cpu_info["model"] = line.split(":")[1].strip()
                        break
                        
                for line in cpuinfo.split("\n"):
                    if "vendor_id" in line:
                        cpu_info["vendor"] = line.split(":")[1].strip()
                        break
                        
                # Essayer d'obtenir la fréquence
                try:
                    frequencies = psutil.cpu_freq()
                    if frequencies:
                        if frequencies.current:
                            cpu_info["frequency_mhz"] = frequencies.current
                        if frequencies.max:
                            cpu_info["max_frequency_mhz"] = frequencies.max
                except Exception:
                    pass
            except Exception:
                pass
        elif platform.system() == "Darwin":  # macOS
            try:
                import subprocess
                # Obtenir le modèle du CPU sur macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                        capture_output=True, text=True, check=True)
                if result.stdout:
                    cpu_info["model"] = result.stdout.strip()
                
                # Obtenir le vendeur du CPU
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.vendor'], 
                                        capture_output=True, text=True, check=True)
                if result.stdout:
                    cpu_info["vendor"] = result.stdout.strip()
                    
                # Obtenir la fréquence
                result = subprocess.run(['sysctl', '-n', 'hw.cpufrequency'], 
                                        capture_output=True, text=True, check=True)
                if result.stdout:
                    try:
                        cpu_info["frequency_mhz"] = int(result.stdout.strip()) / 1000000
                    except ValueError:
                        pass
                
                # Vérifier si MPS (Metal Performance Shaders) est disponible
                if platform.machine() == "arm64":  # Apple Silicon
                    system_info["has_mps"] = True
                else:
                    system_info["has_mps"] = False
            except Exception:
                pass
        elif platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                     r"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0")
                cpu_info["vendor"] = winreg.QueryValueEx(key, "VendorIdentifier")[0]
                cpu_info["model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                try:
                    cpu_info["frequency_mhz"] = winreg.QueryValueEx(key, "~MHz")[0]
                except Exception:
                    pass
                winreg.CloseKey(key)
            except Exception:
                pass
        
        system_info["cpu"] = cpu_info
        
        # Mémoire Information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            "total_bytes": memory.total,
            "free_bytes": memory.available,
            "used_bytes": memory.used,
            "percent_used": memory.percent,
            "swap_total": swap.total,
            "swap_free": swap.free,
            "swap_percent": swap.percent
        }
        
        system_info["memory"] = memory_info
        
        # GPU Information - NVIDIA
        system_info["has_cuda"] = False
        system_info["has_rocm"] = False
        gpu_info = {}
        
        # Essayer avec py3nvml d'abord (plus complet pour NVIDIA)
        if has_nvml:
            try:
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()
                
                if device_count > 0:
                    system_info["has_cuda"] = True
                    
                    # Obtenir des informations sur le premier GPU (généralement utilisé pour l'inférence)
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_info["count"] = device_count
                    
                    # Informations de base sur le GPU
                    gpu_info["name"] = nvml.nvmlDeviceGetName(handle)
                    gpu_info["vendor"] = "NVIDIA"
                    gpu_info["uuid"] = nvml.nvmlDeviceGetUUID(handle)
                    gpu_info["driver_version"] = nvml.nvmlSystemGetDriverVersion()
                    
                    # Mémoire
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info["memory_total"] = mem_info.total
                    gpu_info["memory_free"] = mem_info.free
                    gpu_info["memory_used"] = mem_info.used
                    
                    # Utilisation
                    gpu_info["utilization_percent"] = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    
                    # Température
                    try:
                        gpu_info["temperature_c"] = nvml.nvmlDeviceGetTemperature(
                            handle, nvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        pass
                    
                    # Puissance
                    try:
                        gpu_info["power_usage_watts"] = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        gpu_info["power_limit_watts"] = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    except Exception:
                        pass
                    
                    # Capacité de calcul
                    try:
                        major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
                        gpu_info["compute_capability"] = f"{major}.{minor}"
                    except Exception:
                        pass
                    
                    # Horloge
                    try:
                        gpu_info["clock_mhz"] = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                        gpu_info["memory_clock_mhz"] = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                    except Exception:
                        pass
                    
                    # Nombre de CUDA cores (estimation basée sur compute capability)
                    if "compute_capability" in gpu_info:
                        cc = gpu_info["compute_capability"]
                        cuda_cores = 0
                        
                        # Cette estimation est approximative et ne fonctionne qu'avec certaines architectures NVIDIA
                        if cc.startswith("7."):  # Volta, Turing
                            cuda_cores = 64
                        elif cc.startswith("8."):  # Ampere
                            cuda_cores = 128
                        elif cc.startswith("9."):  # Ada Lovelace, Hopper
                            cuda_cores = 128
                            
                        if cuda_cores > 0:
                            try:
                                sm_count = nvml.nvmlDeviceGetNumSMs(handle)
                                gpu_info["cuda_cores"] = cuda_cores * sm_count
                                
                                # Estimation approximative des TFLOPS
                                if "clock_mhz" in gpu_info and "cuda_cores" in gpu_info:
                                    clock_mhz = gpu_info["clock_mhz"]
                                    cores = gpu_info["cuda_cores"]
                                    gpu_info["tflops"] = (clock_mhz * 2 * cores) / 1000000
                            except Exception:
                                pass
                
                nvml.nvmlShutdown()
            except Exception as e:
                # Si NVML échoue, essayer avec GPUtil
                pass
        
        # Essayer GPUtil si nvml n'a pas fonctionné
        if not system_info["has_cuda"] and has_gputil:
            try:
                gpus = GPUtil.getGPUs()
                
                if gpus:
                    system_info["has_cuda"] = True
                    gpu_info["count"] = len(gpus)
                    
                    # Use first GPU (typically used for inference)
                    gpu = gpus[0]
                    
                    gpu_info["name"] = gpu.name
                    gpu_info["vendor"] = "NVIDIA"  # GPUtil only supports NVIDIA
                    gpu_info["memory_total"] = gpu.memoryTotal * 1024 * 1024  # Convert from MB to bytes
                    gpu_info["memory_free"] = gpu.memoryFree * 1024 * 1024
                    gpu_info["memory_used"] = gpu.memoryUsed * 1024 * 1024
                    gpu_info["utilization_percent"] = gpu.load * 100
                    gpu_info["temperature_c"] = gpu.temperature
                    gpu_info["uuid"] = gpu.uuid
                    gpu_info["device_id"] = gpu.id
            except Exception:
                pass
        
        # Vérifier AMD ROCm (détection simplifiée)
        if not system_info["has_cuda"]:
            try:
                import subprocess
                result = subprocess.run(['rocminfo'], capture_output=True, text=True)
                if result.returncode == 0:
                    system_info["has_rocm"] = True
                    gpu_info["vendor"] = "AMD"
                    
                    # Essayer d'extraire des informations de base de rocminfo
                    output = result.stdout
                    for line in output.splitlines():
                        if "Name:" in line and not "name" in gpu_info:
                            gpu_info["name"] = line.split("Name:")[1].strip()
                        if "Marketing Name:" in line and not "name" in gpu_info:
                            gpu_info["name"] = line.split("Marketing Name:")[1].strip()
            except Exception:
                pass
        
        # Si nous avons des informations GPU, ajoutez-les au système
        if gpu_info:
            system_info["gpu"] = gpu_info
        
        return system_info

    def _get_disk_info(self) -> Dict[str, Any]:
        """Collecter des informations détaillées sur l'utilisation du disque.
        
        Returns:
            Dictionary avec les informations détaillées sur l'utilisation du disque.
        """
        import os
        import shutil
        import platform
        
        disk_info = {}
        
        try:
            # Obtenir des informations sur le disque système
            if platform.system() == "Windows":
                # Sur Windows, utiliser les lecteurs pour lesquels Python est installé
                drive = os.path.splitdrive(sys.executable)[0] or "C:"
                total, used, free = shutil.disk_usage(drive)
                disk_info["system_drive"] = {
                    "path": drive,
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "usage_percent": (used / total) * 100 if total > 0 else 0
                }
            else:
                # Sur Unix/Linux/MacOS, obtenir des informations sur le système de fichiers racine
                total, used, free = shutil.disk_usage("/")
                disk_info["root_fs"] = {
                    "path": "/",
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "usage_percent": (used / total) * 100 if total > 0 else 0
                }
            
            # Obtenir des informations sur le répertoire de travail courant
            cwd = os.getcwd()
            try:
                total, used, free = shutil.disk_usage(cwd)
                disk_info["working_directory"] = {
                    "path": cwd,
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "usage_percent": (used / total) * 100 if total > 0 else 0
                }
            except Exception:
                # Ignorer si nous ne pouvons pas obtenir les informations sur le répertoire de travail
                pass
                
            # Collecter des informations sur le répertoire des modèles d'Ollama
            ollama_models_dir = os.path.expanduser("~/.ollama/models")
            if os.path.exists(ollama_models_dir):
                try:
                    # Taille totale du répertoire des modèles
                    models_size = 0
                    model_count = 0
                    
                    for dirpath, _, filenames in os.walk(ollama_models_dir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            try:
                                models_size += os.path.getsize(fp)
                                if f.endswith(".bin"):  # Fichier de modèle
                                    model_count += 1
                            except:
                                pass
                                
                    # Obtenir l'espace disponible pour le répertoire des modèles
                    total, used, free = shutil.disk_usage(ollama_models_dir)
                    
                    disk_info["ollama_models_directory"] = {
                        "path": ollama_models_dir,
                        "models_total_size_bytes": models_size,
                        "approximate_model_count": model_count,
                        "total_bytes": total,
                        "used_bytes": used,
                        "free_bytes": free,
                        "usage_percent": (used / total) * 100 if total > 0 else 0
                    }
                except Exception as e:
                    logger.warning(f"Erreur lors de la collecte d'informations sur le répertoire des modèles: {e}")
                    
            # Ajouter des métriques agrégées
            disk_info["summary"] = {
                "has_enough_space": True,  # Par défaut à True, à mettre à jour ci-dessous
                "total_free_space_bytes": 0,
                "total_model_storage_bytes": 0
            }
            
            # Mettre à jour le résumé avec les valeurs calculées
            if "root_fs" in disk_info:
                disk_info["summary"]["total_free_space_bytes"] = disk_info["root_fs"]["free_bytes"]
                
                # Vérifier si nous avons suffisamment d'espace (moins de 10% d'espace libre est dangereux)
                if disk_info["root_fs"]["usage_percent"] > 90:
                    disk_info["summary"]["has_enough_space"] = False
                    
            if "ollama_models_directory" in disk_info:
                disk_info["summary"]["total_model_storage_bytes"] = disk_info["ollama_models_directory"]["models_total_size_bytes"]
                
            # Calcul de l'espace disponible pour de nouveaux modèles (estimation prudente)
            if "ollama_models_directory" in disk_info:
                # Prendre 80% de l'espace libre comme espace disponible pour de nouveaux modèles
                # (C'est une estimation prudente pour éviter de remplir complètement le disque)
                available_for_models = disk_info["ollama_models_directory"]["free_bytes"] * 0.8
                disk_info["summary"]["available_for_new_models_bytes"] = int(available_for_models)
                
                # Convertir en GB pour une valeur plus lisible
                disk_info["summary"]["available_for_new_models_gb"] = round(available_for_models / (1024 * 1024 * 1024), 2)
        
        except Exception as e:
            logger.error(f"Erreur lors de la collecte d'informations sur le disque: {e}")
            disk_info["error"] = str(e)
            disk_info["status"] = "error"
        else:
            disk_info["status"] = "ok"
            
        return disk_info

    def _get_ollama_capabilities(self) -> Dict[str, Any]:
        """Collecter des informations détaillées sur les capacités du serveur Ollama.
        
        Returns:
            Dictionary avec les informations complètes sur le serveur Ollama.
        """
        import requests
        import json
        import subprocess
        import platform
        import psutil
        from datetime import datetime
        
        capabilities = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "host_info": {},
            "ollama_info": {},
            "models": [],
            "hardware": {},
            "metrics": {}
        }
        
        try:
            # 1. Informations sur l'hôte
            capabilities["host_info"] = {
                "hostname": platform.node(),
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "memory_total_bytes": psutil.virtual_memory().total,
                "memory_available_bytes": psutil.virtual_memory().available,
                "memory_used_percent": psutil.virtual_memory().percent
            }
            
            # 2. Vérifier si Ollama est en cours d'exécution
            ollama_running = False
            for proc in psutil.process_iter(['name', 'cmdline']):
                if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                    ollama_running = True
                    break
                elif proc.info['cmdline'] and any('ollama' in cmd.lower() for cmd in proc.info['cmdline'] if cmd):
                    ollama_running = True
                    break
            
            capabilities["ollama_info"]["is_running"] = ollama_running
            
            # 3. Obtenir la version d'Ollama (si possible)
            try:
                result = subprocess.run(["ollama", "version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    capabilities["ollama_info"]["version"] = result.stdout.strip()
                    capabilities["status"] = "ok"
                else:
                    capabilities["ollama_info"]["version_error"] = result.stderr.strip()
                    capabilities["status"] = "error"
            except Exception as e:
                capabilities["ollama_info"]["version_error"] = str(e)
                capabilities["status"] = "error"
            
            # 4. Collecter des informations sur les modèles via l'API Ollama
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models_data = response.json().get("models", [])
                    capabilities["models"] = models_data
                    
                    # Calculer la taille totale des modèles
                    total_size = sum(model.get("size", 0) for model in models_data)
                    capabilities["ollama_info"]["total_models_size_bytes"] = total_size
                    capabilities["ollama_info"]["model_count"] = len(models_data)
                else:
                    capabilities["ollama_info"]["models_error"] = f"Erreur HTTP {response.status_code}"
            except Exception as e:
                capabilities["ollama_info"]["models_error"] = str(e)
            
            # 5. Collecter des informations sur le matériel (GPU)
            try:
                # Vérifier si CUDA est disponible via nvidia-smi
                gpu_info = {"gpu_available": False, "devices": []}
                
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], 
                                         capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_info["gpu_available"] = True
                        
                        for i, line in enumerate(result.stdout.strip().split('\n')):
                            if line:
                                parts = [p.strip() for p in line.split(',')]
                                if len(parts) >= 4:
                                    gpu_device = {
                                        "index": i,
                                        "name": parts[0],
                                        "memory_total_mb": float(parts[1]),
                                        "memory_used_mb": float(parts[2]),
                                        "utilization_percent": float(parts[3])
                                    }
                                    gpu_info["devices"].append(gpu_device)
                except Exception:
                    # Si nvidia-smi échoue, essayer avec rocm-smi pour les GPU AMD
                    try:
                        result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            gpu_info["gpu_available"] = True
                            gpu_info["type"] = "amd"
                            # Analyse simplifiée, à améliorer si nécessaire
                            gpu_info["raw_info"] = result.stdout.strip()
                    except Exception:
                        pass
                
                capabilities["hardware"]["gpu"] = gpu_info
            except Exception as e:
                capabilities["hardware"]["gpu_error"] = str(e)
            
            # 6. Métriques de performance
            try:
                capabilities["metrics"] = {
                    "cpu_usage_percent": psutil.cpu_percent(interval=0.5),
                    "memory_usage_percent": psutil.virtual_memory().percent,
                    "swap_usage_percent": psutil.swap_memory().percent if hasattr(psutil, "swap_memory") else None,
                    "disk_io": {
                        "read_count": psutil.disk_io_counters().read_count,
                        "write_count": psutil.disk_io_counters().write_count,
                        "read_bytes": psutil.disk_io_counters().read_bytes,
                        "write_bytes": psutil.disk_io_counters().write_bytes
                    } if hasattr(psutil, "disk_io_counters") and psutil.disk_io_counters() else {},
                    "network": {
                        "bytes_sent": psutil.net_io_counters().bytes_sent,
                        "bytes_recv": psutil.net_io_counters().bytes_recv,
                        "packets_sent": psutil.net_io_counters().packets_sent,
                        "packets_recv": psutil.net_io_counters().packets_recv
                    } if hasattr(psutil, "net_io_counters") and psutil.net_io_counters() else {}
                }
            except Exception as e:
                capabilities["metrics_error"] = str(e)
            
            # 7. Métriques spécifiques aux LLMs (temps de génération moyen, etc.)
            # Ces métriques pourraient être maintenues par l'application et ajoutées ici
            capabilities["llm_metrics"] = {
                "available": False,
                "note": "Les métriques LLM détaillées ne sont pas encore implémentées"
            }
            
        except Exception as e:
            capabilities["status"] = "error"
            capabilities["error"] = str(e)
        
        return capabilities


def check_ollama_running(ollama_host: str = "http://localhost:11434") -> bool:
    """Vérifie si le serveur Ollama est en cours d'exécution.
    
    Args:
        ollama_host: URL du serveur Ollama (par défaut: http://localhost:11434)
        
    Returns:
        True si le serveur est en cours d'exécution, False sinon
    """
    try:
        # Essayer d'accéder à l'API Ollama pour vérifier son état
        response = requests.get(f"{ollama_host}/api/version", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"Ollama server check failed: {str(e)}")
        return False

def ensure_ollama_running(ollama_host: str = "http://localhost:11434", 
                        timeout: int = 30,
                        ollama_command: str = "ollama") -> bool:
    """Assure que le serveur Ollama est en cours d'exécution, le démarre si nécessaire.
    
    Args:
        ollama_host: URL du serveur Ollama (par défaut: http://localhost:11434)
        timeout: Temps d'attente maximum en secondes
        ollama_command: Commande pour démarrer Ollama (par défaut: ollama)
        
    Returns:
        True si le serveur est en cours d'exécution après l'opération, False sinon
    """
    # Vérifier d'abord si le serveur est déjà en cours d'exécution
    if check_ollama_running(ollama_host):
        logger.debug("Ollama server is already running")
        return True
    
    # Essayer de démarrer le serveur uniquement sur localhost
    parsed_url = urlparse(ollama_host)
    hostname = parsed_url.netloc.split(':')[0]
    
    if hostname not in ('localhost', '127.0.0.1', '0.0.0.0', '::1'):
        logger.warning(f"Cannot start Ollama server on remote host: {hostname}")
        return False
    
    try:
        # Démarrer le serveur Ollama en arrière-plan
        logger.info("Starting Ollama server...")
        
        if platform.system() == "Windows":
            # Sur Windows, utiliser CREATE_NEW_PROCESS_GROUP pour détacher le processus
            process = subprocess.Popen(
                f"{ollama_command} serve",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # Sur Unix/Linux/macOS, utiliser nohup
            process = subprocess.Popen(
                f"nohup {ollama_command} serve > /dev/null 2>&1 &",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Attendre que le serveur démarre
        start_time = time.time()
        while time.time() - start_time < timeout:
            if check_ollama_running(ollama_host):
                logger.info("Ollama server started successfully")
                return True
            time.sleep(1)
        
        logger.error(f"Ollama server did not start within {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Error starting Ollama server: {str(e)}")
        return False