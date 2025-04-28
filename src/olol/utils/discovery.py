"""Discovery utilities for OLOL servers and proxies."""

import json
import logging
import socket
import struct
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default multicast group and port for discovery
DEFAULT_MULTICAST_GROUP = "224.0.0.251"
DEFAULT_MULTICAST_PORT = 5355
DEFAULT_DISCOVERY_INTERVAL = 30  # seconds
DEFAULT_TIMEOUT = 2  # seconds

# Message types
MSG_TYPE_PROXY_ANNOUNCE = "proxy-announce"
MSG_TYPE_SERVER_ANNOUNCE = "server-announce"
MSG_TYPE_SERVER_REGISTER = "server-register"
MSG_TYPE_PROXY_ACK = "proxy-ack"


class DiscoveryService:
    """Service for automatic discovery of OLOL proxies and servers."""
    
    def __init__(self, 
                 service_id: Optional[str] = None,
                 service_type: str = "server",
                 multicast_group: str = DEFAULT_MULTICAST_GROUP,
                 multicast_port: int = DEFAULT_MULTICAST_PORT,
                 service_port: int = 50052,
                 extra_info: Optional[Dict[str, Any]] = None,
                 preferred_interface: Optional[str] = None) -> None:
        """Initialize the discovery service.
        
        Args:
            service_id: Unique ID for this service, defaults to a UUID
            service_type: Type of service ('server' or 'proxy')
            multicast_group: Multicast group address for discovery
            multicast_port: Multicast port for discovery
            service_port: Port where this service is running
            extra_info: Additional information to include in announcements
            preferred_interface: IP address of the preferred network interface
        """
        self.service_id = service_id or str(uuid.uuid4())
        self.service_type = service_type
        self.multicast_group = multicast_group
        self.multicast_port = multicast_port
        self.service_port = service_port
        self.extra_info = extra_info or {}
        self.preferred_interface = preferred_interface
        
        # For connection latency tracking
        self.connection_latencies: Dict[Tuple[str, int], float] = {}
        
        # For tracking discovered services
        self.discovered_services: Dict[str, Dict[str, Any]] = {}
        self.services_lock = threading.Lock()
        
        # For callbacks when services are discovered
        self.discovery_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Sockets for sending/receiving
        self.unicast_socket: Optional[socket.socket] = None
        self.multicast_socket: Optional[socket.socket] = None
        self.ipv6_supported = self._check_ipv6_support()
        
        # Thread control
        self.running = False
        self.announcement_thread: Optional[threading.Thread] = None
        self.listen_thread: Optional[threading.Thread] = None
        
    def _check_ipv6_support(self) -> bool:
        """Check if IPv6 is supported on this system.
        
        Returns:
            True if IPv6 is supported, False otherwise
        """
        try:
            # Try to create an IPv6 socket
            s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            s.close()
            logger.debug("IPv6 support detected")
            return True
        except (socket.error, OSError):
            logger.debug("IPv6 not supported")
            return False
        
    def register_discovery_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback for when services are discovered.
        
        Args:
            callback: Function called with (service_id, service_info) when a service is discovered
        """
        self.discovery_callbacks.append(callback)
        
    def start(self) -> None:
        """Start the discovery service."""
        if self.running:
            return
            
        self.running = True
        
        try:
            # Create unicast socket based on IPv6 support
            if self.ipv6_supported:
                try:
                    # Create dual-stack socket (IPv4 + IPv6)
                    self.unicast_socket = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
                    self.unicast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    # Ensure it works for both IPv4 and IPv6 (disable IPv6-only)
                    self.unicast_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                    logger.debug("Using dual-stack IPv4/IPv6 socket for unicast")
                except (socket.error, OSError) as e:
                    logger.warning(f"Failed to create dual-stack socket: {e}, falling back to IPv4")
                    self.unicast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self.unicast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            else:
                # IPv6 not supported, use IPv4 only
                self.unicast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.unicast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Create multicast socket for discovery (always IPv4 for multicast compatibility)
            self.multicast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.multicast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to the multicast port
            self.multicast_socket.bind(('', self.multicast_port))
            
            # Join the multicast group
            mreq = struct.pack('4sL', socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
            self.multicast_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            # Set timeout for receive operations
            self.multicast_socket.settimeout(DEFAULT_TIMEOUT)
            
            # Start listener thread
            self.listen_thread = threading.Thread(target=self._listen_for_messages, daemon=True)
            self.listen_thread.start()
            
            # Start announcement thread if we're a server
            if self.service_type in ("server", "proxy"):
                self.announcement_thread = threading.Thread(target=self._send_announcements, daemon=True)
                self.announcement_thread.start()
                
            logger.info(f"Discovery service started for {self.service_type} with ID {self.service_id} (IPv6: {'Supported' if self.ipv6_supported else 'Not supported'})")
        except Exception as e:
            logger.error(f"Failed to start discovery service: {e}")
            self.stop()
    
    def stop(self) -> None:
        """Stop the discovery service."""
        self.running = False
        
        # Close sockets
        if self.unicast_socket:
            try:
                self.unicast_socket.close()
            except:
                pass
            self.unicast_socket = None
            
        if self.multicast_socket:
            try:
                self.multicast_socket.close()
            except:
                pass
            self.multicast_socket = None
        
        # Wait for threads to terminate
        if self.announcement_thread and self.announcement_thread.is_alive():
            self.announcement_thread.join(timeout=2)
            
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2)
            
        logger.info("Discovery service stopped")
        
    def _create_message(self, msg_type: str, target_id: Optional[str] = None) -> dict:
        """Create a discovery message.
        
        Args:
            msg_type: Type of message
            target_id: Optional target service ID
            
        Returns:
            Message dictionary
        """
        # Get all IP addresses for this host
        hostname = socket.gethostname()
        interface_ips = self._get_all_interface_ips()
        
        # Always include localhost in interfaces
        if "127.0.0.1" not in interface_ips:
            interface_ips.append("127.0.0.1")
            
        # For IPv6, include ::1 if IPv6 is supported
        if self.ipv6_supported and "::1" not in interface_ips:
            interface_ips.append("::1")
        
        # Primary IP (for backwards compatibility)
        # Prefer a non-localhost address if available
        non_local_ips = [ip for ip in interface_ips 
                         if not ip.startswith("127.") and 
                            ip != "::1" and 
                            not ip.startswith("fe80:")]
        
        primary_ip = non_local_ips[0] if non_local_ips else interface_ips[0] if interface_ips else "127.0.0.1"
            
        # Build connection endpoints list with ports
        connection_endpoints = []
        
        # Add all interfaces with their ports
        for ip in interface_ips:
            # For IPv6, use proper [IPv6]:port format
            if ':' in ip and not ip.startswith('localhost'):
                connection_endpoints.append(f"[{ip}]:{self.service_port}")
            else:
                connection_endpoints.append(f"{ip}:{self.service_port}")
                
        # Create message with enhanced routing information
        message = {
            "msg_type": msg_type,
            "service_id": self.service_id,
            "service_type": self.service_type,
            "hostname": hostname,
            "ip": primary_ip,  # Primary IP for backward compatibility
            "interfaces": interface_ips,  # All possible interfaces 
            "connection_endpoints": connection_endpoints,  # Full connection strings
            "port": self.service_port,
            "source_port": self.multicast_port,  # Include source port for NAT traversal
            "timestamp": time.time()
        }
        
        if target_id:
            message["target_id"] = target_id
            
        # Add extra information
        if self.extra_info:
            message["capabilities"] = self.extra_info
            
        return message
        
    def _send_to_multicast(self, multicast_address: str, message: Dict) -> bool:
        """Send a message to a multicast address.
        
        Args:
            multicast_address: The multicast group address
            message: The message to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.unicast_socket:
            return False
            
        try:
            # Try to resolve hostname to IP if it's not already an IP address
            try:
                if not all(c.isdigit() or c == '.' for c in multicast_address):
                    multicast_ip = socket.gethostbyname(multicast_address)
                else:
                    multicast_ip = multicast_address
            except socket.gaierror:
                # If resolution fails, use the original address
                multicast_ip = multicast_address
                
            # Encode message
            encoded_message = json.dumps(message).encode('utf-8')
            
            # Try to send over IPv4
            try:
                self.unicast_socket.sendto(encoded_message, (multicast_ip, self.multicast_port))
                return True
            except Exception as ipv4_error:
                logger.debug(f"IPv4 multicast to {multicast_ip} failed: {ipv4_error}")
                
                # If IPv6 is supported, try that as fallback
                if self.ipv6_supported and ':' not in multicast_ip:
                    # Try using IPv6 multicast address
                    try:
                        # ff02::1 is all nodes on the local link
                        self.unicast_socket.sendto(encoded_message, ("ff02::1", self.multicast_port, 0, 0))
                        return True
                    except Exception as ipv6_error:
                        logger.debug(f"IPv6 multicast fallback failed: {ipv6_error}")
                        
            # If we got here, both methods failed
            return False
        except Exception as e:
            logger.debug(f"Error sending to multicast {multicast_address}: {e}")
            return False
    
    def _send_announcements(self) -> None:
        """Periodically send service announcements."""
        while self.running:
            try:
                if self.service_type == "proxy":
                    # Proxies announce their presence to servers
                    message = self._create_message(MSG_TYPE_PROXY_ANNOUNCE)
                else:
                    # Servers announce their presence to proxies
                    message = self._create_message(MSG_TYPE_SERVER_ANNOUNCE)
                    
                # Send to multicast group - try multiple methods to avoid the "Address family not supported" error
                sent_successfully = False
                error_messages = []
                
                # Try each of these multicast addresses in order
                multicast_addresses = [
                    self.multicast_group,  # Default (224.0.0.251)
                    "224.0.0.1",           # All hosts on the local network
                    "239.255.255.250",     # SSDP discovery address
                    "127.0.0.1"            # Last resort - localhost
                ]
                
                # Try each address until one works
                for address in multicast_addresses:
                    try:
                        if self._send_to_multicast(address, message):
                            sent_successfully = True
                            break
                    except Exception as e:
                        error_messages.append(f"{address}: {str(e)}")
                
                # Log if all methods failed
                if not sent_successfully:
                    error_summary = "; ".join(error_messages)
                    logger.warning(f"Failed to send announcement to any multicast group: {error_summary}")
                    
                # Also try to directly connect to any known proxies if we're a server
                if self.service_type == "server":
                    with self.services_lock:
                        for service_id, service_info in self.discovered_services.items():
                            if (service_info.get("service_type") == "proxy" and 
                                "ip" in service_info and "port" in service_info):
                                target_ip = service_info["ip"]
                                target_port = self.multicast_port
                                
                                # Check if this is an IPv6 address
                                is_ipv6 = ':' in target_ip and not target_ip.startswith('localhost')
                                
                                # Skip if IPv6 not supported
                                if is_ipv6 and not self.ipv6_supported:
                                    logger.debug(f"Skipping IPv6 proxy {target_ip} - IPv6 not supported")
                                    continue
                                
                                # Try to resolve hostname to IP if it's not already an IP address
                                # This handles potential hosts file entries
                                if not is_ipv6 and not all(c.isdigit() or c == '.' for c in target_ip):
                                    try:
                                        # Get the IP address from hostname
                                        resolved_ip = socket.gethostbyname(target_ip)
                                        if resolved_ip != target_ip:
                                            logger.debug(f"Resolved {target_ip} to {resolved_ip}")
                                            target_ip = resolved_ip
                                    except socket.gaierror as e:
                                        logger.warning(f"Could not resolve hostname {target_ip}: {e}")
                                        continue
                                    
                                try:
                                    # Send a registration message directly to the proxy
                                    register_msg = self._create_message(
                                        MSG_TYPE_SERVER_REGISTER,
                                        target_id=service_id
                                    )
                                    
                                    if self.unicast_socket:
                                        try:
                                            # For IPv6, we need to wrap the address in a tuple with flow and scope IDs
                                            if is_ipv6:
                                                self.unicast_socket.sendto(
                                                    json.dumps(register_msg).encode('utf-8'),
                                                    (target_ip, target_port, 0, 0)
                                                )
                                            else:
                                                self.unicast_socket.sendto(
                                                    json.dumps(register_msg).encode('utf-8'),
                                                    (target_ip, target_port)
                                                )
                                        except Exception as send_err:
                                            logger.warning(f"Failed to send to {target_ip}: {send_err}")
                                except Exception as e:
                                    logger.warning(f"Failed to register with proxy {service_id}: {e}")
                    
                # Wait before sending next announcement
                time.sleep(DEFAULT_DISCOVERY_INTERVAL)
            except Exception as e:
                logger.error(f"Error sending announcement: {e}")
                time.sleep(5)  # Shorter delay on error
                
    def _listen_for_messages(self) -> None:
        """Listen for discovery messages."""
        while self.running and self.multicast_socket:
            try:
                # Receive a message
                try:
                    data, addr = self.multicast_socket.recvfrom(4096)
                    sender_ip, sender_port = addr
                except socket.timeout:
                    continue
                    
                # Parse the message
                try:
                    message = json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    continue
                    
                # Skip our own messages
                if message.get("service_id") == self.service_id:
                    continue
                    
                # Process based on message type
                msg_type = message.get("msg_type", "")
                
                if msg_type == MSG_TYPE_PROXY_ANNOUNCE and self.service_type == "server":
                    # Server discovered a proxy
                    self._handle_proxy_announcement(message, sender_ip)
                    
                elif msg_type == MSG_TYPE_SERVER_ANNOUNCE and self.service_type == "proxy":
                    # Proxy discovered a server
                    self._handle_server_announcement(message, sender_ip)
                    
                elif msg_type == MSG_TYPE_SERVER_REGISTER and self.service_type == "proxy":
                    # Server is registering with this proxy
                    self._handle_server_registration(message, sender_ip)
                    
                elif msg_type == MSG_TYPE_PROXY_ACK and self.service_type == "server":
                    # Proxy acknowledged registration
                    self._handle_proxy_acknowledgement(message)
                    
            except Exception as e:
                logger.error(f"Error in discovery listener: {e}")
                    
    def _get_all_interface_ips(self) -> List[str]:
        """Get all IP addresses for all network interfaces.
        
        Returns:
            List of IP addresses
        """
        ips = []
        
        # If preferred interface is specified, prioritize it
        if self.preferred_interface:
            ips.append(self.preferred_interface)
            
        try:
            # Try to use netifaces if available (more reliable)
            try:
                import netifaces
                for interface in netifaces.interfaces():
                    # Get IPv4 addresses
                    for link in netifaces.ifaddresses(interface).get(netifaces.AF_INET, []):
                        address = link.get('addr')
                        if address and address != '127.0.0.1' and address not in ips:
                            ips.append(address)
                    
                    # Get IPv6 addresses if supported
                    if self.ipv6_supported:
                        for link in netifaces.ifaddresses(interface).get(netifaces.AF_INET6, []):
                            address = link.get('addr')
                            # Filter out link-local addresses (fe80::) as they're not globally routable
                            if (address and 
                                address != '::1' and 
                                not address.startswith('fe80:') and 
                                address not in ips):
                                # Remove scope ID if present (%eth0, etc.)
                                if '%' in address:
                                    address = address.split('%')[0]
                                ips.append(address)
            except ImportError:
                # Fall back to socket approach
                import socket
                hostname = socket.gethostname()
                
                # Get hostname IPv4
                try:
                    host_ip = socket.gethostbyname(hostname)
                    if host_ip and host_ip != '127.0.0.1' and host_ip not in ips:
                        ips.append(host_ip)
                except:
                    pass
                    
                # Try to get all IPs including hostname variants
                try:
                    fqdn = socket.getfqdn()
                    # Get both IPv4 and IPv6 addresses
                    fqdn_ips = socket.getaddrinfo(fqdn, None)
                    for ip_info in fqdn_ips:
                        ip_family = ip_info[0]
                        ip = ip_info[4][0]
                        
                        # Handle IPv4
                        if ip_family == socket.AF_INET:
                            if ip and ip != '127.0.0.1' and ip not in ips:
                                ips.append(ip)
                        
                        # Handle IPv6 if supported
                        elif self.ipv6_supported and ip_family == socket.AF_INET6:
                            if (ip and 
                                ip != '::1' and 
                                not ip.startswith('fe80:') and 
                                ip not in ips):
                                # Remove scope ID if present
                                if '%' in ip:
                                    ip = ip.split('%')[0]
                                ips.append(ip)
                except:
                    pass
                
                # Try to directly get IPv4 interfaces
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # Doesn't matter which address, just needs to be routable
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                    if ip and ip != '127.0.0.1' and ip not in ips:
                        ips.append(ip)
                except:
                    pass
                finally:
                    s.close()
                    
                # Try to directly get IPv6 interfaces if supported
                if self.ipv6_supported:
                    try:
                        s6 = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
                        # Connect to IPv6 Google DNS
                        s6.connect(("2001:4860:4860::8888", 80))
                        ip = s6.getsockname()[0]
                        if (ip and 
                            ip != '::1' and 
                            not ip.startswith('fe80:') and 
                            ip not in ips):
                            # Remove scope ID if present
                            if '%' in ip:
                                ip = ip.split('%')[0]
                            ips.append(ip)
                        s6.close()
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Error getting network interfaces: {e}")
            
        # Always add localhost as last resort
        if not ips:
            ips.append("127.0.0.1")
            
        # Filter out IPv6 addresses if not supported on this host
        if not self.ipv6_supported:
            ips = [ip for ip in ips if ':' not in ip]
            
        logger.debug(f"Identified interfaces: {ips}")
        return ips
        
    def _measure_connection_latency(self, host: str, port: int) -> Optional[float]:
        """Measure connection latency to a host:port.
        
        Args:
            host: Target hostname or IP
            port: Target port
            
        Returns:
            Latency in milliseconds, or None if connection failed
        """
        start_time = time.time()
        
        # Determine if this is an IPv6 address
        is_ipv6 = ':' in host and not host.startswith('localhost')
        
        # Skip IPv6 addresses if not supported
        if is_ipv6 and not self.ipv6_supported:
            logger.debug(f"Skipping IPv6 address {host} as IPv6 is not supported")
            return None
            
        try:
            # Choose socket type based on address format
            if is_ipv6:
                # IPv6 address
                s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            else:
                # IPv4 address or hostname
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
            s.settimeout(1.0)  # 1 second timeout
            
            # Connect appropriate to address type
            if is_ipv6:
                # For IPv6, we need to specify flowinfo and scopeid
                s.connect((host, port, 0, 0))
            else:
                s.connect((host, port))
                
            s.close()
            latency = (time.time() - start_time) * 1000  # Convert to ms
            return latency
        except Exception as e:
            logger.debug(f"Failed to connect to {host}:{port} - {e}")
            return None
            
    def _try_all_interfaces(self, message: Dict, sender_ip: str) -> Dict[str, float]:
        """Try to connect to all interfaces and endpoints provided in a message.
        
        Args:
            message: Message containing interface information
            sender_ip: IP address of the sender
            
        Returns:
            Dictionary mapping reachable interfaces to their latency
        """
        results = {}
        port = message.get("port", self.service_port)
        source_port = message.get("source_port", self.multicast_port)
        
        # Get sender's source port from packet - useful for NAT traversal
        packet_source_endpoint = f"{sender_ip}:{source_port}"
        logger.debug(f"Packet source endpoint: {packet_source_endpoint}")
        
        # First try the connection_endpoints list which has proper formatting for both IPv4 and IPv6
        connection_endpoints = message.get("connection_endpoints", [])
        
        if connection_endpoints:
            logger.debug(f"Trying {len(connection_endpoints)} connection endpoints from message")
            for endpoint in connection_endpoints:
                if endpoint in results:
                    continue  # Already tried
                
                try:
                    # Parse endpoint into host and port
                    if endpoint.startswith('['):
                        # IPv6 address with port: [IPv6]:port
                        host = endpoint[1:endpoint.rindex(']')]
                        port_str = endpoint[endpoint.rindex(']')+2:]
                        try:
                            endpoint_port = int(port_str)
                        except ValueError:
                            endpoint_port = port
                    else:
                        # IPv4 address or hostname with port: host:port
                        try:
                            host, port_str = endpoint.rsplit(':', 1)
                            try:
                                endpoint_port = int(port_str)
                            except ValueError:
                                endpoint_port = port
                        except ValueError:
                            # No port in endpoint
                            host = endpoint
                            endpoint_port = port
                            
                    # Skip if this is an IPv6 address but IPv6 is not supported
                    is_ipv6 = ':' in host and not host.startswith('localhost')
                    if is_ipv6 and not self.ipv6_supported:
                        logger.debug(f"Skipping IPv6 endpoint {endpoint} as IPv6 is not supported")
                        continue
                    
                    # Try to connect
                    latency = self._measure_connection_latency(host, endpoint_port)
                    if latency is not None:
                        results[endpoint] = latency
                        logger.debug(f"Successfully connected to {endpoint} with latency {latency:.2f}ms")
                except Exception as e:
                    logger.debug(f"Error parsing/connecting to endpoint {endpoint}: {e}")
        
        # If no explicit connection_endpoints or none worked, try the legacy approach
        if not results:
            logger.debug("No working connection endpoints found, trying individual interfaces")
            
            # Try the primary IP first
            primary_ip = message.get("ip")
            if primary_ip:
                latency = self._measure_connection_latency(primary_ip, port)
                if latency is not None:
                    results[primary_ip] = latency
                    logger.debug(f"Successfully connected to primary IP {primary_ip}:{port} with latency {latency:.2f}ms")
            
            # Try all interfaces
            interfaces = message.get("interfaces", [])
            if not interfaces and primary_ip:
                interfaces = [primary_ip]
                
            for ip in interfaces:
                endpoint = f"{ip}:{port}"
                if ip in results or endpoint in results:
                    continue  # Already tried
                    
                # Skip if this is an IPv6 address but IPv6 is not supported
                is_ipv6 = ':' in ip and not ip.startswith('localhost')
                if is_ipv6 and not self.ipv6_supported:
                    logger.debug(f"Skipping IPv6 interface {ip} as IPv6 is not supported")
                    continue
                
                latency = self._measure_connection_latency(ip, port)
                if latency is not None:
                    endpoint_key = f"{ip}:{port}"
                    results[endpoint_key] = latency
                    logger.debug(f"Successfully connected to interface {endpoint_key} with latency {latency:.2f}ms")
        
        # Special case: If we're finding a tunneled server, try the sender IP with both the declared port and source port
        if sender_ip:
            # Try with the declared service port
            sender_endpoint = f"{sender_ip}:{port}"
            if sender_endpoint not in results:
                latency = self._measure_connection_latency(sender_ip, port)
                if latency is not None:
                    results[sender_endpoint] = latency
                    logger.debug(f"Successfully connected to sender {sender_endpoint} with latency {latency:.2f}ms")
            
            # Also try with the source port (for NAT traversal)
            sender_source_endpoint = f"{sender_ip}:{source_port}"
            if sender_source_endpoint != sender_endpoint and sender_source_endpoint not in results:
                latency = self._measure_connection_latency(sender_ip, source_port)
                if latency is not None:
                    results[sender_source_endpoint] = latency
                    logger.debug(f"Successfully connected to sender source port {sender_source_endpoint} with latency {latency:.2f}ms")
                
        # If no connections worked, use sender IP as a last resort with infinite latency
        if not results and sender_ip:
            sender_endpoint = f"{sender_ip}:{port}"
            results[sender_endpoint] = float('inf')  # Infinite latency, but we'll try it
            logger.debug(f"No connections successful, using sender {sender_endpoint} as last resort")
            
        return results
            
    def _handle_proxy_announcement(self, message: Dict, sender_ip: str) -> None:
        """Handle a proxy announcement message.
        
        Args:
            message: Announcement message
            sender_ip: IP address of the sender
        """
        service_id = message.get("service_id")
        if not service_id:
            return
        
        # Try all interfaces to find the best one
        reachable_interfaces = self._try_all_interfaces(message, sender_ip)
        
        if reachable_interfaces:
            # Use the interface with lowest latency
            best_ip = min(reachable_interfaces.items(), key=lambda x: x[1])[0]
            latency = reachable_interfaces[best_ip]
            
            # Update the message with best IP
            message["reachable_ips"] = list(reachable_interfaces.keys())
            message["best_ip"] = best_ip
            message["latency_ms"] = latency
            
            # If original IP isn't reachable, update it
            if "ip" not in message or message["ip"] not in reachable_interfaces:
                message["ip"] = best_ip
                
            logger.info(f"Proxy {service_id} reachable at {best_ip} with latency {latency:.2f}ms")
        else:
            # If no interface is reachable, just use sender_ip
            message["ip"] = sender_ip
            message["not_reachable"] = True
            logger.warning(f"Proxy {service_id} is not directly reachable")
            
        # Update the discovered services
        with self.services_lock:
            self.discovered_services[service_id] = message
            
        # Call discovery callbacks
        for callback in self.discovery_callbacks:
            try:
                callback(service_id, message)
            except Exception as e:
                logger.error(f"Error in discovery callback: {e}")
                
        # Send a registration message to the proxy
        try:
            register_msg = self._create_message(MSG_TYPE_SERVER_REGISTER, target_id=service_id)
            if self.unicast_socket:
                self.unicast_socket.sendto(
                    json.dumps(register_msg).encode('utf-8'),
                    (message["ip"], self.multicast_port)
                )
        except Exception as e:
            logger.warning(f"Failed to register with proxy {service_id}: {e}")
            
    def _handle_server_announcement(self, message: Dict, sender_ip: str) -> None:
        """Handle a server announcement message.
        
        Args:
            message: Announcement message
            sender_ip: IP address of the sender
        """
        service_id = message.get("service_id")
        if not service_id:
            return
        
        # Try all interfaces to find the best one
        reachable_interfaces = self._try_all_interfaces(message, sender_ip)
        
        if reachable_interfaces:
            # Use the interface with lowest latency
            best_ip = min(reachable_interfaces.items(), key=lambda x: x[1])[0]
            latency = reachable_interfaces[best_ip]
            
            # Update the message with best IP
            message["reachable_ips"] = list(reachable_interfaces.keys())
            message["best_ip"] = best_ip
            message["latency_ms"] = latency
            
            # If original IP isn't reachable, update it
            if "ip" not in message or message["ip"] not in reachable_interfaces:
                message["ip"] = best_ip
                
            logger.info(f"Server {service_id} reachable at {best_ip} with latency {latency:.2f}ms")
        else:
            # If no interface is reachable, just use sender_ip
            message["ip"] = sender_ip
            message["not_reachable"] = True
            logger.warning(f"Server {service_id} is not directly reachable")
            
        # Update the discovered services
        with self.services_lock:
            self.discovered_services[service_id] = message
            
        # Call discovery callbacks
        for callback in self.discovery_callbacks:
            try:
                callback(service_id, message)
            except Exception as e:
                logger.error(f"Error in discovery callback: {e}")
                
        # Send an acknowledgement to the server using best IP if available
        try:
            ack_msg = self._create_message(MSG_TYPE_PROXY_ACK, target_id=service_id)
            if self.unicast_socket:
                # Try to send to the best IP first if available
                target_ip = message.get("best_ip", message.get("ip", sender_ip))
                
                # Handle IPv6 addresses properly
                is_ipv6 = ':' in target_ip and not target_ip.startswith('localhost')
                
                # Skip IPv6 addresses if not supported
                if is_ipv6 and not self.ipv6_supported:
                    logger.warning(f"Skipping acknowledgement to IPv6 address {target_ip} as IPv6 is not supported")
                else:
                    try:
                        # For IPv6, we need to wrap the address in a tuple with flow and scope IDs
                        if is_ipv6:
                            self.unicast_socket.sendto(
                                json.dumps(ack_msg).encode('utf-8'),
                                (target_ip, self.multicast_port, 0, 0)
                            )
                        else:
                            self.unicast_socket.sendto(
                                json.dumps(ack_msg).encode('utf-8'),
                                (target_ip, self.multicast_port)
                            )
                    except Exception as e:
                        logger.warning(f"Failed to send acknowledgement to {target_ip}: {e}")
                
                # If we have multiple reachable IPs, try them all for redundancy
                for ip in message.get("reachable_ips", []):
                    if ip != target_ip:  # Skip the one we already sent to
                        is_ipv6 = ':' in ip and not ip.startswith('localhost')
                        
                        if is_ipv6 and not self.ipv6_supported:
                            continue  # Skip IPv6 if not supported
                            
                        try:
                            # For IPv6, we need to wrap the address in a tuple with flow and scope IDs
                            if is_ipv6:
                                self.unicast_socket.sendto(
                                    json.dumps(ack_msg).encode('utf-8'),
                                    (ip, self.multicast_port, 0, 0)
                                )
                            else:
                                self.unicast_socket.sendto(
                                    json.dumps(ack_msg).encode('utf-8'),
                                    (ip, self.multicast_port)
                                )
                        except Exception:
                            # Ignore errors for redundant sends
                            pass
        except Exception as e:
            logger.warning(f"Failed to acknowledge server {service_id}: {e}")
            
    def _handle_server_registration(self, message: Dict, sender_ip: str) -> None:
        """Handle a server registration message.
        
        Args:
            message: Registration message
            sender_ip: IP address of the sender
        """
        if message.get("target_id") != self.service_id:
            # This registration is not for us
            return
            
        service_id = message.get("service_id")
        if not service_id:
            return
            
        # Update IP if it's missing or incorrect
        if "ip" not in message or not message["ip"]:
            message["ip"] = sender_ip
            
        # Update the discovered services
        with self.services_lock:
            self.discovered_services[service_id] = message
            
        # Call discovery callbacks
        for callback in self.discovery_callbacks:
            try:
                callback(service_id, message)
            except Exception as e:
                logger.error(f"Error in discovery callback: {e}")
                
        # Send an acknowledgement to the server
        try:
            ack_msg = self._create_message(MSG_TYPE_PROXY_ACK, target_id=service_id)
            target_ip = message["ip"]
            target_port = self.multicast_port
            
            # Check if this is an IPv6 address
            is_ipv6 = ':' in target_ip and not target_ip.startswith('localhost')
            
            # Skip if IPv6 not supported
            if is_ipv6 and not self.ipv6_supported:
                logger.warning(f"Skipping acknowledgement to IPv6 address {target_ip} as IPv6 is not supported")
                return
            
            # Try to resolve hostname to IP if it's not already an IP address
            # This handles potential hosts file entries
            if not is_ipv6 and not all(c.isdigit() or c == '.' for c in target_ip):
                try:
                    # Get the IP address from hostname
                    resolved_ip = socket.gethostbyname(target_ip)
                    if resolved_ip != target_ip:
                        logger.debug(f"Resolved {target_ip} to {resolved_ip}")
                        target_ip = resolved_ip
                except socket.gaierror as e:
                    logger.warning(f"Could not resolve hostname {target_ip} for acknowledgement: {e}")
                    # Continue with original IP
            
            if self.unicast_socket:
                try:
                    # For IPv6, we need to wrap the address in a tuple with flow and scope IDs
                    if is_ipv6:
                        self.unicast_socket.sendto(
                            json.dumps(ack_msg).encode('utf-8'),
                            (target_ip, target_port, 0, 0)
                        )
                    else:
                        self.unicast_socket.sendto(
                            json.dumps(ack_msg).encode('utf-8'),
                            (target_ip, target_port)
                        )
                except Exception as send_err:
                    logger.warning(f"Failed to send acknowledgement to {target_ip}: {send_err}")
        except Exception as e:
            logger.warning(f"Failed to acknowledge server {service_id}: {e}")
            
    def _handle_proxy_acknowledgement(self, message: Dict) -> None:
        """Handle a proxy acknowledgement message.
        
        Args:
            message: Acknowledgement message
        """
        if message.get("target_id") != self.service_id:
            # This acknowledgement is not for us
            return
            
        service_id = message.get("service_id")
        if not service_id:
            return
            
        # Update the discovered services
        with self.services_lock:
            # Mark this proxy as confirmed
            if service_id in self.discovered_services:
                self.discovered_services[service_id]["confirmed"] = True
                logger.info(f"Registration confirmed by proxy {service_id}")
            else:
                # This is a new proxy
                self.discovered_services[service_id] = message
                self.discovered_services[service_id]["confirmed"] = True
                logger.info(f"Discovered new proxy {service_id}")
                
        # Call discovery callbacks
        for callback in self.discovery_callbacks:
            try:
                callback(service_id, message)
            except Exception as e:
                logger.error(f"Error in discovery callback: {e}")
                
    def get_discovered_proxies(self) -> List[Dict[str, Any]]:
        """Get list of discovered proxies.
        
        Returns:
            List of proxy information dictionaries
        """
        with self.services_lock:
            return [
                service_info for service_id, service_info in self.discovered_services.items()
                if service_info.get("service_type") == "proxy"
            ]
            
    def get_discovered_servers(self) -> List[Dict[str, Any]]:
        """Get list of discovered servers.
        
        Returns:
            List of server information dictionaries
        """
        with self.services_lock:
            return [
                service_info for service_id, service_info in self.discovered_services.items()
                if service_info.get("service_type") == "server"
            ]


def get_capabilities_info() -> Dict[str, Any]:
    """Get information about this machine's capabilities.
    
    Returns:
        Dictionary with capability information
    """
    capabilities = {}
    
    # Get OS information
    try:
        import platform
        capabilities["os"] = platform.system()
        capabilities["os_version"] = platform.version()
        capabilities["architecture"] = platform.machine()
    except Exception as e:
        logger.debug(f"Failed to get OS info: {e}")
        
    # Get CPU information
    try:
        import multiprocessing
        capabilities["cpu_cores"] = multiprocessing.cpu_count()
    except Exception as e:
        logger.debug(f"Failed to get CPU info: {e}")
        
    # Get memory information
    try:
        import psutil
        mem = psutil.virtual_memory()
        capabilities["memory"] = mem.total
        capabilities["memory_available"] = mem.available
    except Exception as e:
        logger.debug(f"Failed to get memory info: {e}")
        
    # Get GPU information
    capabilities["gpu"] = []
    
    # Try CUDA
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(i)
                capabilities["gpu"].append({
                    "type": "cuda",
                    "device_id": i,
                    "name": device.name,
                    "memory": device.total_memory,
                    "compute_capability": f"{device.major}.{device.minor}"
                })
    except Exception as e:
        logger.debug(f"Failed to get CUDA info: {e}")
        
    # Try ROCm
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            for i in range(torch.hip.device_count()):
                device = torch.hip.get_device_properties(i)
                capabilities["gpu"].append({
                    "type": "rocm",
                    "device_id": i,
                    "name": device.name,
                    "memory": device.total_memory
                })
    except Exception as e:
        logger.debug(f"Failed to get ROCm info: {e}")
        
    # Try Metal (Apple)
    try:
        import torch
        if (hasattr(torch, 'backends') and 
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()):
            capabilities["gpu"].append({
                "type": "metal",
                "device_id": 0,
                "name": "Apple Metal GPU"
            })
    except Exception as e:
        logger.debug(f"Failed to get Metal info: {e}")
        
    # Get Ollama information
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            data = response.json()
            capabilities["ollama_version"] = data.get("version")
            
            # Try to get status for more info
            try:
                status_resp = requests.get("http://localhost:11434/api/status", timeout=2)
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    # Add GPU info if available
                    if "gpus" in status_data:
                        capabilities["gpus"] = [
                            {
                                "name": gpu.get("name", ""),
                                "memory": gpu.get("memory", ""),
                                "compute": gpu.get("compute", ""),
                            }
                            for gpu in status_data["gpus"]
                        ]
            except Exception as status_err:
                logger.debug(f"Failed to get Ollama status: {status_err}")
    except Exception as e:
        logger.debug(f"Failed to get Ollama info: {e}")
        
        # Try another method
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout:
                capabilities["ollama_version"] = result.stdout.strip()
        except Exception as e2:
            logger.debug(f"Failed to get Ollama version from CLI: {e2}")
        
    return capabilities