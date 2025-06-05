import uvicorn
import argparse
import socket
import ipaddress
from .server import app
from .segment import global_config
import torch.cuda

def get_args():
    parser = argparse.ArgumentParser(description="ITK-SNAP deep learning segmentation server configuration")

    # Port number, default to 8911 unless it's a well-known port
    parser.add_argument(
        "--port", '-p',
        type=int, 
        default=8911,  
        help="Port number for the server (default: 8911)"
    )

    # Optional hostname
    parser.add_argument(
        "--host", '-H',
        type=str,
        default="0.0.0.0",
        help="Hostname for the server (default: 0.0.0.0)"
    )

    # Location for Hugging Face models
    parser.add_argument(
        "--models-path", '-m',
        type=str,
        help="Location where to download deep learning models"
    )

    # Torch device selection
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device to use (default: 'cuda' if available, else 'cpu')"
    )
    
    # Skip verification
    parser.add_argument("-k", "--insecure", 
                        action="store_true",
                        help="Skip HTTPS certificate verification")
    
    # Use ngrok for tunneling
    parser.add_argument("-N", "--ngrok", 
                        action="store_true",
                        help="Create a public URL using ngrok for the server. An NGROK_AUTHTOKEN environment variable must be set to use this feature.")
    
    # Force color output
    parser.add_argument("--use-colors",
                        action="store_true",
                        help="Force colored output in the terminal")

    # Run initial setup, including downloading models, bot not the server
    parser.add_argument("--setup-only",
                        action="store_true",
                        help="Run initial setup, including downloading models, but not starting the server")

    return parser.parse_args()

def print_gpu_info():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        gpu_index = torch.cuda.current_device()
        print(f"    Using GPU {gpu_index}: {gpu_name}")
    else:
        print(f"    No GPU available, using CPU.")

def print_banner(host: str, port: int):
    print(f'***************** ITK-SNAP Deep Learning Extensions Server ******************')

    print_gpu_info()
    urls = []
    
    # Get all network interfaces
    hostname = socket.gethostname()
    local_ips = socket.getaddrinfo(hostname, None)
    unique_ips = set(ip[-1][0] for ip in local_ips)
    
    for ip in unique_ips:
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_loopback:
            urls.append([ip, port, True])
            urls.append(['localhost', port, True])
        else:
            urls.append([ip, port, False])
            urls.append([socket.getfqdn(ip), port, False])

    usort = sorted(set(tuple(x) for x in urls))
    print(f'    Use one of the following settings in ITK-SNAP to connect to this server:')
    for url in list([x for x in usort if x[2] is False]):
        print(f'        Server: {url[0]:40s}  Port: {url[1]}')
    for url in list([x for x in usort if x[2] is True]):
        print(f'        Server: {url[0]:40s}  Port: {url[1]}  †')
    print(f'        †: only works if ITK-SNAP is running on the same computer')

    print(f'******************************************************************************')


def print_banner_ngrok(url: str):
    print(f'***************** ITK-SNAP Deep Learning Extensions Server ******************')

    print_gpu_info()
    
    # Remove https:// from the URL
    if url.startswith("https://"):
        url = url[8:]
    
    print(f'    Use one of the following settings in ITK-SNAP to connect to this server:')
    print(f'        Server: {url}  Port: {443}')

    print(f'******************************************************************************')

def get_access_urls(host: str, port: int):

    """Generate a list of URLs based on the system's network interfaces."""
    urls = []
    
    if host in ["0.0.0.0", "::"]:
        # Get all network interfaces
        hostname = socket.gethostname()
        local_ips = socket.getaddrinfo(hostname, None)
        unique_ips = set(ip[-1][0] for ip in local_ips)
        
        for ip in unique_ips:
            urls.append([ip, port])

    else:
        urls.append([host, port])

    return urls


if __name__ == "__main__":
    args = get_args()
    global_config.device = args.device
    global_config.hf_models_path = args.models_path
    global_config.https_verify = args.insecure
    
    # Special mode to run setup only
    if args.setup_only:
        from .segment import SegmentSession
        print(f'Running setup only, downloading models to {args.models_path}')
        segment_session = SegmentSession(config=global_config)
        print(f'Setup complete. Models are available at {segment_session.model_path}')
        exit(0)
        
    # Create an ngrok session if requested
    if args.ngrok:
        
        # Check if the NGROK_AUTHTOKEN environment variable is set
        import os
        if 'NGROK_AUTHTOKEN' not in os.environ:
            print('NGROK_AUTHTOKEN environment variable is not set. Please set it to use ngrok.')
            print(' - Sign up for an account: https://dashboard.ngrok.com/signup ')
            print(' - Obtain your authtoken: https://dashboard.ngrok.com/get-started/your-authtoken ')
            print(' - Set environment variable NGROK_AUTHTOKEN to your authtoken value.')
            exit(1)
        
        import ngrok
        try:
            listener = ngrok.forward(args.port, authtoken_from_env=True)
        except ValueError as e:
            print(f'Error starting ngrok: {e}')
            exit(1)
        print_banner_ngrok(listener.url())
        
    else:

        # Print how to access the server
        print_banner(args.host, port=args.port)
    
    if args.use_colors:
        uvicorn.run(app, host=args.host, port=args.port, use_colors=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
