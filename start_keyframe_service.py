#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
启动KeyframeGen服务的脚本
"""

import os
import sys
import yaml
import argparse
import Pyro5.nameserver
from keyframe_service import start_keyframe_service

def main():
    parser = argparse.ArgumentParser(description='启动KeyframeGen服务')
    parser.add_argument('--config', type=str, default='keyframe_service_config.yaml', help='配置文件路径')
    parser.add_argument('--host', type=str, help='服务主机地址，覆盖配置文件')
    parser.add_argument('--port', type=int, help='服务端口，覆盖配置文件')
    parser.add_argument('--start-nameserver', action='store_true', help='是否启动名称服务器')
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        config = {
            'service': {'host': '0.0.0.0', 'port': 9090},
            'nameserver': {'host': 'localhost', 'port': 9090},
            'logging': {'level': 'INFO', 'file': 'keyframe_service.log'}
        }
    
    # 命令行参数覆盖配置文件
    host = args.host if args.host else config['service']['host']
    port = args.port if args.port else config['service']['port']
    
    # 启动名称服务器
    if args.start_nameserver:
        print(f"启动名称服务器在 {config['nameserver']['host']}:{config['nameserver']['port']}...")
        try:
            # 启动名称服务器的进程
            import subprocess
            import threading
            
            def run_nameserver():
                subprocess.run([
                    sys.executable, 
                    "-m", 
                    "Pyro5.nameserver", 
                    "--host", 
                    config['nameserver']['host'],
                    "--port",
                    str(config['nameserver']['port'])
                ])
            
            # 在单独的线程中启动名称服务器
            ns_thread = threading.Thread(target=run_nameserver)
            ns_thread.daemon = True
            ns_thread.start()
            
            # 等待名称服务器启动
            print("等待名称服务器启动...")
            import time
            time.sleep(2)
            
        except Exception as e:
            print(f"启动名称服务器失败: {str(e)}")
            print("请手动启动名称服务器: python -m Pyro5.nameserver")
            return
    
    # 启动KeyframeGen服务
    print(f"启动KeyframeGen服务在 {host}:{port}...")
    start_keyframe_service(host=host, port=port)

if __name__ == "__main__":
    main()
