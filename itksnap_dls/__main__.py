import uvicorn
import argparse
import socket
from .server import app, server_startup
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

    return parser.parse_args()


def get_access_urls(host: str, port: int):
    """Generate a list of URLs based on the system's network interfaces."""
    urls = []
    
    if host in ["0.0.0.0", "::"]:
        # Get all network interfaces
        hostname = socket.gethostname()
        local_ips = socket.getaddrinfo(hostname, None)
        unique_ips = set(ip[-1][0] for ip in local_ips)
        
        for ip in unique_ips:
            protocol = "http"  # Change to https if needed
            urls.append(f"{protocol}://{ip}:{port}")

    else:
        urls.append(f"http://{host}:{port}")

    return urls


if __name__ == "__main__":
    args = get_args()
    global_config.device = args.device
    global_config.hf_models_path = args.models_path
    global_config.https_verify = args.insecure

    # Print how to access the server
    print(f'************ ITK-SNAP Deep Learning Extensions Server ************')
    urls = get_access_urls(args.host, port=args.port)
    print(f'Use one of the following URLs to access the server from ITK-SNAP:')
    for url in urls:
        print(f'    {url}')
    print(f'******************************************************************')

    server_startup()
    uvicorn.run(app, host=args.host, port=args.port)
