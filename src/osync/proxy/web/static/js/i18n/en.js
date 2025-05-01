/**
 * Ollama Sync - English translations
 */

// English translations
window.I18n = window.I18n || {};
window.I18n.translations = window.I18n.translations || {};

window.I18n.translations['en'] = {
    // Application general
    "app": {
        "name": "Ollama Sync",
        "copyright": "Ollama Sync © 2025",
        "version": "Version 1.0.0"
    },
    
    // Navigation
    "nav": {
        "dashboard": "Dashboard",
        "models": "Models",
        "servers": "Servers",
        "health": "Health",
        "queue": "Queue",
        "settings": "Settings",
        "playground": "Playground",
        "terminal": "Terminal",
        "log": "Logs",
        "swagger": "API Docs",
        "api_documentation": "API Documentation",
        "language": "Language"
    },
    
    // Sidebar
    "sidebar": {
        "collapse": "Collapse",
        "expand": "Expand"
    },
    
    // Notifications
    "notifications": {
        "title": "Notifications",
        "new_server": "New Server Detected",
        "server_offline": "Server Offline",
        "high_load": "High Load",
        "model_downloaded": "Model Downloaded",
        "just_now": "Just now",
        "see_all": "See All Notifications",
        "success": "Success",
        "error": "Error",
        "warning": "Warning",
        "info": "Information"
    },
    
    // Dashboard
    "dashboard": {
        "title": "System Dashboard",
        "system_stats": "System Statistics",
        "active_servers": "Active Servers",
        "total_models": "Total Models",
        "active_requests": "Active Requests",
        "request_rate": "Request Rate",
        "cpu_usage": "CPU Usage",
        "memory_usage": "Memory Usage",
        "disk_usage": "Disk Usage",
        "network_usage": "Network Usage",
        "recent_activity": "Recent Activity",
        "quick_actions": "Quick Actions"
    },
    
    // Models
    "models": {
        "title": "Models Management",
        "available_models": "Available Models",
        "model_name": "Model Name",
        "size": "Size",
        "modified": "Modified",
        "quantization": "Quantization",
        "download": "Download",
        "delete": "Delete",
        "pull": {
            "title": "Download Model",
            "submit": "Download",
            "progress": "Downloading...",
            "progress_msg": "Downloading {{model}}...",
            "complete": "Download Complete",
            "success": "{{model}} has been downloaded successfully"
        },
        "details": "Model Information",
        "info": {
            "general": "General Information",
            "title_for": "Information: {{name}}"
        },
        "parameters": "parameters",
        "server_location": "Server Location",
        "loading": "Loading models...",
        "status": {
            "available": "Available",
            "unavailable": "Unavailable"
        },
        "add": {
            "name": "Model Name",
            "server": "Target Server"
        },
        "name": {
            "placeholder": "e.g. llama2, mistral, etc.",
            "help": "Enter the model name to download from the Ollama catalog."
        },
        "server": {
            "select": "Select a server",
            "all": "All servers",
            "help": "Select the server where you want to download this model."
        },
        "insecure": "Allow Insecure Sources",
        "none_available": "No models available. Use the \"Download Model\" button to add one.",
        "filter": {
            "by": "Filter by",
            "all": "All Models",
            "local": "Local Models",
            "downloaded": "Downloaded Models",
            "placeholder": "Search for a model..."
        },
        "table": {
            "name": "Name",
            "size": "Size:",
            "servers": "Available Servers:",
            "parameters": "Parameters:",
            "modified": "Last Updated:",
            "family": "Family",
            "quantization": "Quantization"
        },
        "test": "Test",
        "copy_prompt": "Copy Prompt",
        "distribute": {
            "title": "Distribution in Progress",
            "progress": "Distributing {{model}} to all servers...",
            "complete": "Distribution Complete",
            "success": "{{model}} has been distributed successfully"
        },
        "delete": {
            "title": "Deletion in Progress",
            "confirm": "Confirm Deletion",
            "confirm_text": "Are you sure you want to delete the model",
            "warning": "This action will delete the model from all servers where it is installed.",
            "progress": "Deleting...",
            "progress_msg": "Deleting {{model}}...",
            "complete": "Deletion Complete",
            "success": "{{model}} has been deleted successfully"
        },
        "availability": "Availability",
        "prompt_template": "Prompt Template",
        "default_params": "Default Parameters",
        "prompt_copied": "Prompt template copied to clipboard",
        "prompt_copy_error": "Unable to copy the template"
    },
    
    // Servers
    "servers": {
        "title": "Server Management",
        "available_servers": "Available Servers",
        "hostname": "Hostname",
        "address": "Address",
        "status": "Status",
        "load": "Load",
        "online": "Online",
        "offline": "Offline",
        "add_server": "Add Server",
        "remove": "Remove",
        "server_details": "Server Details",
        "uptime": "Uptime",
        "cpu": "CPU",
        "memory": "Memory",
        "models_hosted": "Models Hosted",
        "cluster_status": "Cluster Status",
        "active": "Active Servers",
        "manage": "Manage Servers",
        "none_available": "No Servers",
        "status": {
            "proxy_active": "Proxy Active"
        },
        "table": {
            "name": "Server",
            "status": "Status"
        }
    },
    
    // Health
    "health": {
        "title": "System Health",
        "system_status": "System Status",
        "server_status": "Server Status",
        "response_time": "Response Time",
        "uptime": "Uptime",
        "last_checked": "Last Checked",
        "healthy": "Healthy",
        "degraded": "Degraded",
        "unhealthy": "Unhealthy",
        "check_now": "Check Now",
        "refresh": {
            "auto": "Auto Refresh"
        }
    },
    
    // Settings
    "settings": {
        "title": "System Settings",
        "general": "General",
        "appearance": "Appearance",
        "network": "Network",
        "language": {
            "title": "Language",
            "selector": "Language Selection",
            "current": "Current Language",
            "changed": "Language has been changed successfully"
        },
        "theme": "Theme",
        "dark_mode": "Dark Mode",
        "light_mode": "Light Mode",
        "auto_discovery": "Auto Discovery",
        "save": "Save Changes",
        "reset": "Reset to Defaults",
        "enable": "Enable",
        "disable": "Disable"
    },
    
    // Queue
    "queue": {
        "title": "Request Queue",
        "current_queue": "Current Queue",
        "request_id": "Request ID",
        "model": "Model",
        "timestamp": "Timestamp",
        "status": "Status",
        "action": "Action",
        "pending": "Pending",
        "processing": "Processing",
        "completed": "Completed",
        "failed": "Failed",
        "cancel": "Cancel",
        "retry": "Retry"
    },
    
    // Playground
    "playground": {
        "title": "Model Playground",
        "select_model": "Select Model",
        "parameters": "Parameters",
        "temperature": "Temperature",
        "max_tokens": "Max Tokens",
        "top_p": "Top P",
        "prompt": "Enter your prompt here...",
        "generate": "Generate",
        "stop": "Stop",
        "response": "Response",
        "copy": "Copy",
        "clear": "Clear",
        "save": "Save"
    },
    
    // Terminal
    "terminal": {
        "title": "Terminal Interface",
        "welcome": "Welcome to Ollama Sync Terminal",
        "help": "Type 'help' for a list of commands",
        "command": "Command",
        "output": "Output",
        "clear": "Clear Terminal"
    },
    
    // Common actions/buttons
    "action": {
        "add": "Add",
        "edit": "Edit",
        "delete": "Delete",
        "cancel": "Cancel",
        "save": "Save",
        "refresh": "Refresh",
        "close": "Close",
        "confirm": "Confirm",
        "back": "Back",
        "next": "Next",
        "active": "Active Refresh"
    },
    
    // Success messages
    "success": {
        "copied": "Success",
        "saved": "Saved successfully",
        "updated": "Updated successfully",
        "deleted": "Deleted successfully"
    },
    
    // Error messages
    "error": {
        "generic": "Error",
        "required_field": "Please fill in all required fields",
        "connection": "Connection error",
        "not_found": "Not found",
        "server_error": "Server error",
        "permission": "Permission error"
    },
    
    // Pagination
    "pagination": {
        "previous": "Previous",
        "next": "Next",
        "showing": "Showing {{start}} to {{end}} of {{total}} entries"
    },
    
    // API documentation
    "api": {
        "title": "API Documentation",
        "description": "Interactive API documentation for Ollama Sync",
        "endpoints": "Endpoints",
        "models": "Models",
        "try_it": "Try It",
        "response": "Response",
        "status": "Status"
    }
};