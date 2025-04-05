import argparse
from .visual_html import visualize
from rich import console

console = console.Console()

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--main_folder", type=str, required=True)
parser.add_argument('-n', "--nodes_num", type=int, default=500,help="nodes number")
args = parser.parse_args()

console.print(f"Visualizing {args.main_folder} with nodes number {args.nodes_num}")
visualize(args.main_folder, args.nodes_num)

