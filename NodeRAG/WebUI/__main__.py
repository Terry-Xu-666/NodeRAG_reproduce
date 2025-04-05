import argparse
import os
import sys
import streamlit.web.cli as stcli

def main():
    parser = argparse.ArgumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("-f", "--main_folder", type=str, help="main folder path")
    args = parser.parse_args()

    
    app_path = os.path.join(current_dir, "app.py")
    
    # 使用 -- 传递参数给 Streamlit
    sys.argv = [
        "streamlit", 
        "run", 
        app_path,
        "--",
        f"--main_folder={args.main_folder}"
    ]
    
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()