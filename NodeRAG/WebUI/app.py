import streamlit as st
import time
from typing import List,Tuple
import yaml
import os
import json
import sys

from NodeRAG.utils import LazyImport

NG = LazyImport('NodeRAG','NodeRag')
NGSearch = LazyImport('NodeRAG','NodeSearch')
NGConfig = LazyImport('NodeRAG','NodeConfig')



# init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'settings' not in st.session_state:
    st.session_state.settings = {}
    st.session_state.settings['relevant_info'] = 'On'
if 'indices' not in st.session_state:
    st.session_state.indices = {}
if 'main_folder' not in st.session_state:
    st.session_state.main_folder = None


args = sys.argv


config_path = None
for arg in args:
    if arg.startswith('--main_folder='):
        main_folder = arg.split('=')[1]
st.session_state.original_config_path = os.path.join(main_folder, 'Node_config.yaml')
st.session_state.web_ui_config_path = os.path.join(main_folder, 'web_ui_config.yaml')



class State_Observer:
    
    def __init__(self,build_status):
        self.build_status = build_status
    
    def update(self,state):
        self.build_status.status(f"🔄 Building Status: {state}")
    
    def reset(self,total_tasks:List[str],desc:str=""):
        if isinstance(total_tasks, list):
            task_str = "\n".join([f"  └─ {task}" for task in total_tasks])
            st.markdown(f"🔄 Building Status: {desc}\nTasks:\n{task_str}")
            time.sleep(2)

    def close(self):
        self.build_status.empty()
        
        
def load_config(path):
    """Load the config from the config file"""
    with open(path, 'r') as file:
        all_config = yaml.safe_load(file)
        st.session_state.config = all_config['config']
        st.session_state.model_config = all_config['model_config']
        st.session_state.embedding_config = all_config['embedding_config']

def all_config():
    """Get all the config from the session state"""
    return {
        'config': st.session_state.config,
        'model_config': st.session_state.model_config,
        'embedding_config': st.session_state.embedding_config
    }
    
def save_config(path):
    """Save the config to the config file"""
    with open(path, 'w') as file:
        yaml.dump(all_config(), file)
    

def display_header():
    """Display the header section with title and description"""
    # Create two columns for title and expander
    col1, col2 = st.columns([0.5, 0.5])  # Numbers represent the width ratio of the columns

    # Put title in first column
    with col1:
        st.title('NodeRAG')

    # Put expander in second column
    with col2:
        st.markdown('<div style="margin-top: 26px;"></div>', unsafe_allow_html=True)
        with st.expander("What is NodeRAG?"):
            st.write('NodeRAG is a graph-based retrieval-augmented generation (RAG) framework that structures knowledge as a heterogeneous graph to enhance retrieval precision and multi-hop reasoning.')

def display_chat_history():
    """Display the chat history from session state"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Show relevant info for assistant messages
            if message["role"] == "assistant":
                if st.session_state.config['relevant_info'] == 'On':
                    if message.get('relevant_info') is not None:
                        Relevant_Information(message['relevant_info'])
# Define different background colors for alternating items
def display_retrieval_list(relevant_list:List[Tuple[str,str]]):
    bg_colors = [
        "rgba(255, 235, 238, 0.3)", # Light red
        "rgba(227, 242, 253, 0.3)", # Light blue  
        "rgba(232, 245, 233, 0.3)", # Light green
        "rgba(255, 243, 224, 0.3)", # Light orange
        "rgba(243, 229, 245, 0.3)"  # Light purple
    ]
    
    # Display each item with alternating background colors
    for i, item in enumerate(relevant_list):
        # Use modulo to cycle through colors
        bg_color = bg_colors[i % len(bg_colors)]
        
        # Add stronger border and increased opacity for better visibility
        st.markdown(
            f"""
            <div style='
                background-color: {bg_color}; 
                padding: 12px;
                margin: 8px 0;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                position: relative;
            '>
                <div style='
                    position: absolute;
                    top: 4px;
                    left: 8px;
                    font-size: 0.8em;
                    color: rgba(0,0,0,0.6);
                '>
                    {item[1]}
                </div>
                <div style='
                    margin-top: 16px;
                '>
                    {item[0]}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
                        
def add_message(role: str, user_input: str):
    """Add a message to the chat history"""
    
    with st.chat_message(role):
        
        # Only show Relevant_Information for assistant messages
        if role == "assistant":
            # Create placeholders for status and message
            status_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            # Show retrieval status
            with status_placeholder.status("Retrieving relevant information..."):
               
        
                searched = st.session_state.settings['search_engine'].search(user_input)

                # Generate and store relevant info
                if st.session_state.settings['relevant_info'] == 'On':
                    if searched.retrieved_list is not None:
                        display_retrieval_list(searched.retrieved_list)
            
            # Show generation status
            with status_placeholder.status("Generating response..."):
                content = st.session_state.settings['search_engine'].stream_answer(user_input,searched.structured_prompt)
                for chunk in content:
                    for char in chunk:
                        full_response += char
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)  # Simulate typing delay
                message_placeholder.markdown(full_response)
            if st.session_state.config['relevant_info'] == 'On':
                st.session_state.messages.append({"role": "assistant", "content":full_response , "relevant_info":searched.retrieved_list})
            else:
                st.session_state.messages.append({"role": "assistant", "content":full_response})
            
            with status_placeholder.status("✅ Retrieval and generation completed"):
                pass
                
            
        else:
            st.write(user_input)
            
        st.session_state.messages.append({"role": role, "content": user_input})
    
def handle_user_input():
    """Handle user input and generate assistant response"""
    if user_input := st.chat_input('Enter your text here:'):
        # Handle user message
        add_message("human", user_input)
        
        # Get and handle assistant response
        add_message("assistant", user_input)
        
def check_building_status(placeholder):
    """Check the building status"""
    with placeholder.status("Checking Building Status"): 
            if os.path.exists(os.path.join(st.session_state.config['main_folder'], 'info/state.json')):
                with open(os.path.join(st.session_state.config['main_folder'], 'info/state.json'), 'r') as f:
                    state = json.load(f)
                st.markdown(f"🔄 Building Status: {state['Current_state']}")
                return True
            else:
                st.markdown("🔹 Building Status: Not Built")
                return False


def sidebar():
    """Display the left sidebar with user input and assistant response"""
    with st.sidebar:
        st.title('Building Status')
        Build_Status = st.empty()
        Building = st.empty()
        check_building_status(Build_Status)
        
            
        if st.button("Build/Update",key="start_building"):
            state_observer = State_Observer(Building)
            state_observer.reset(total_tasks=["1. Document Processing", "2. Text Processing", "3. Graph Processing", "4. Attribute Processing", "5. Embedding Processing", "6. Summary Processing", "7. HNSW Processing","8. Finished"],desc="Building the NodeRAG")
            state_controller = NG(NGConfig(all_config()),web_ui=True)
            state_controller.add_observer(state_observer)
            state_controller.run()
            state_observer.close()
            
        if check_building_status(Build_Status):
            Enable_Search = st.toggle("Search Engine", value=True)
            
            if Enable_Search and not st.session_state.settings.get('engine_running'):
                st.session_state.settings['search_engine'] = NGSearch(NGConfig(all_config()))
                st.session_state.settings['engine_running'] = True
                st.write("Search Engine is running")
                if not st.session_state.indices:
                    st.session_state.indices = json.load(open(os.path.join(st.session_state.config['main_folder'], 'info/indices.json'), 'r'))
                
            elif not Enable_Search and st.session_state.settings.get('engine_running'):
                st.session_state.settings['engine_running'] = False
                st.session_state.settings['search_engine'] = None
                st.write("Search Engine is stopping")

            else:
                pass
            
        if 'Enable_Search' in locals() and Enable_Search:
            with st.expander("📑 Available Indices", expanded=False):
                if st.session_state.indices:
                    for key, value in st.session_state.indices.items():
                        st.markdown(f"**{key.replace('_',' ')}**: {value}")
                else:
                    st.markdown("No indices available")
            
            
            
            
                
        st.title("Settings")
        
        # Get current working directory as default folder
        
        # RAG Build Settings
        with st.expander("🔧 RAG Settings", expanded=False):
            # Basic Settings
            st.markdown("Main Folder: " + st.session_state.config['main_folder'])
            
            new_folder = st.text_input("Enter folder path:",key="main_folder")
            
            if new_folder:
                if os.path.exists(new_folder):
                    new_folder = new_folder.strip().strip('"\'')
                    st.session_state.config['main_folder'] = new_folder
                else:
                    st.error("Invalid folder path")
            
            st.session_state.config['language'] = st.selectbox(
                "Language",
                ["English", "Chinese"],
                index=["English", "Chinese"].index(st.session_state.config['language']),
                help="Processing language"
            )
            
            st.session_state.config['docu_type'] = st.selectbox(
                "Document Type",
                ["mixed", "md", "txt", "docx"],
                index=["mixed", "md", "txt", "docx"].index(st.session_state.config['docu_type']),
                help="Type of documents to process"
            )

            # Chunking Settings
            st.session_state.config['chunk_size'] = st.slider(
                "Chunk Size", 
                min_value=800,
                max_value=2000,
                value= st.session_state.config['chunk_size'],
                step=50,
                help="Size of text chunks for processing"
            )
            
            st.session_state.config['embedding_batch_size'] = st.number_input(
                "Embedding Batch Size",
                min_value=1,
                max_value=100,
                value= st.session_state.config['embedding_batch_size'],
                step=5,
                help="Number of embeddings to process in one batch"
            )


            # HNSW Index Settings
            st.subheader("HNSW Index Settings")
        
            st.session_state.config['dim'] = st.number_input(
                "Dimension",
                min_value=256,
                max_value=2048,
                value= st.session_state.config['dim'],
                help="Dimension of the embedding"
            )
            st.session_state.config['m'] = st.number_input(
                "M Parameter",
                min_value=5,
                max_value=100,
                value= st.session_state.config['m'],
                help="HNSW M parameter (max number of connections per layer)"
            )
            st.session_state.config['ef'] = st.number_input(
                "EF Parameter",
                min_value=50,
                max_value=500,
                value= st.session_state.config['ef'],
                help="HNSW ef parameter (size of dynamic candidate list)"
            )
            st.session_state.config['m0'] = st.radio(
                "M0 Parameter",
                [None],
                index=0,
                help="HNSW m0 parameter (number of bi-directional links)"
            )

            # Summary and Search Settings
            st.subheader("Summary and Search Settings")
        
        
            st.session_state.config['Hcluster_size'] = st.slider(
                "Hcluster Size",
                min_value=39,
                max_value=80,
                value=st.session_state.config['Hcluster_size'],
                step=1,
                help="Size of High level elements cluster"
            )
                

            

        # Model and Embedding Settings
        
        with st.expander("🤖 Model & Embedding Settings", expanded=False):
            st.subheader("Model settings")
            st.session_state.model_config['service_provider'] = st.selectbox(
                "Service Provider",
                ["openai",'gemini'],
                index=["openai",'gemini'].index(st.session_state.model_config['service_provider']),
                help="AI service provider"
            )
            if st.session_state.model_config['service_provider'] == 'openai':
                st.session_state.model_config['model_name'] = st.selectbox(
                    "Language Model",
                    ["gpt-4o-mini","gpt-4o"],
                    help="Select the language model to use"
                )
            elif st.session_state.model_config['service_provider'] == 'gemini':
                st.session_state.model_config['model_name'] = st.selectbox(
                    "Language Model",
                    ["gemini-2.0-flash-lite-preview-02-05"],
                    help="Select the language model to use"
                )
            
            st.markdown(f'api_keys: {st.session_state.model_config["api_keys"][:10] + "..."}')
            model_keys = st.text_input("Enter API Key:",key="model_keys")
            if model_keys:
                st.session_state.model_config['api_keys'] = model_keys.strip().strip('"\'')
                
            st.session_state.model_config['temperature'] = st.slider(
                "Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.model_config['temperature']),
                step=0.05,
                help="Temperature for model generation"
            )
            st.session_state.model_config['max_tokens'] = st.number_input(
                "Model Max Tokens",
                min_value=500,
                max_value=10000,
                value=int(st.session_state.model_config['max_tokens']),
                step=100,
                help="Maximum number of tokens for model generation"
            )
            st.session_state.model_config['rate_limit'] = st.number_input(
                "API Rate Limit (requests at a time)",
                min_value=1,
                max_value=50,
                value=int(st.session_state.model_config['rate_limit']),
                step=1,
                help="Rate limit for API calls"
            )
            
            # Embedding settings
            st.subheader("Embedding settings")
            st.session_state.embedding_config['service_provider'] = st.selectbox(
                "Embedding Provider",
                ["openai_embedding","gemini_embedding"],
                index=["openai_embedding","gemini_embedding"].index(st.session_state.embedding_config['service_provider']),
                help="Embedding service provider"
            )
            if st.session_state.embedding_config['service_provider'] == 'openai_embedding':
                st.session_state.embedding_config['embedding_model_name'] = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-3-small", "text-embedding-3-large"],
                    help="Model used for generating embeddings"
                )
            elif st.session_state.embedding_config['service_provider'] == 'gemini_embedding':
                st.session_state.embedding_config['embedding_model_name'] = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-004"],
                    help="Model used for generating embeddings"
                )
            st.markdown(f'api_keys: {st.session_state.embedding_config["api_keys"][:10] + "..."}')
            embedding_keys = st.text_input("Enter API Key:",key="embedding_keys")
            if embedding_keys:
                st.session_state.embedding_config['api_keys'] = embedding_keys.strip().strip('"\'')
            
            # Rate limits
            st.session_state.embedding_config['rate_limit'] = st.number_input(
                "API Rate Limit (requests/second)",
                min_value=1,
                max_value=50,
                value=st.session_state.embedding_config['rate_limit'],
                step=1,
                help="Rate limit for API calls"
            )
            
                    
        
        
        # File upload menu
        with st.expander("📁 Document Upload", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload your documents",
                accept_multiple_files=True,
                type=['txt', 'doc', 'docx','md']
            )
            
            input_folder = os.path.join(st.session_state.config['main_folder'], 'input')
            if uploaded_files: 
                if show_confirmation_dialog(f"Are you sure you want to upload file to {input_folder}?"):
                    os.makedirs(input_folder, exist_ok=True)
                    for file in uploaded_files:
                        file_path = os.path.join(input_folder, file.name)
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())
                    st.write("Files uploaded successfully")
                
            if os.path.exists(input_folder):
                input_files = os.listdir(input_folder)
                if input_files:
                    st.markdown("### 📄 Files in Input Folder")
                    for file in input_files:
                        st.markdown(f"<div style='margin-left:20px;'><i>📝 {file}</i></div>", unsafe_allow_html=True)
                else:
                    st.write("Input folder is empty")
                    
        # Settings menu
        with st.expander("🔍 Search Settings", expanded=False):
            st.session_state.config['relevant_info'] = st.selectbox(
                'Relevant Information', 
                ['On', 'Off'],
                index=0 
            )
            # Search settings
            
            st.session_state.config['unbalance_adjust'] = st.checkbox(
                "Unbalance  Adjust",
                value=True,
                help="Whether to adjust for unbalanced data"
            )
        
            st.session_state.config['cross_node'] = st.number_input(
                "Cross Node Number",
                min_value=1,
                max_value=50,
                value=st.session_state.config['cross_node'],
                step=1,
                help="Number of cross node"
            )
            
            st.session_state.config['Enode'] = st.number_input(
                "Entity Node",
                min_value=1,
                max_value=50,
                value=st.session_state.config['Enode'],
                step=1,
                help="Number of entity node"
            )
            
            st.session_state.config['Rnode'] = st.number_input(
                "Relation Node",
                min_value=1,
                max_value=50,
                value=st.session_state.config['Rnode'],
                step=1,
                help="Number of relation node"
            )
            
            st.session_state.config['Hnode'] = st.number_input(
                "High Level Node",
                min_value=1,
                max_value=50,
                value=st.session_state.config['Hnode'],
                step=1,
                help="Number of high level node"
            )
            
            st.session_state.config['HNSW_results'] = st.number_input(
                "HNSW Results",
                min_value=1,
                max_value=50,
                value=st.session_state.config['HNSW_results'],
                step=1,
                help="Number of top results to return"
            )
            
            st.session_state.config['ppr_alpha'] = st.slider(
                "PPR Alpha",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.config['ppr_alpha'],
                step=0.01,
                help="Alpha for PPR"
            )
            st.session_state.config['ppr_max_iter'] = st.number_input(
                "PPR Max Iter",
                min_value=1,
                max_value=100,
                value=st.session_state.config['ppr_max_iter'],
                step=1,
                help="Maximum number of iterations for PPR"
            )
            st.session_state.config['similarity_weight'] = st.slider(
                "Similarity Weight",
                min_value=1.0,
                max_value=3.0,
                value=float(st.session_state.config['similarity_weight']),
                step=0.1,
                help="Weight for similarity"
            )
            st.session_state.config['accuracy_weight'] = st.slider(
                "Accuracy Weight",
                min_value=1.0,
                max_value=3.0,
                value=float(st.session_state.config['accuracy_weight']),
                step=0.5,
                help="Weight for accuracy"
            )
        # Save config button
        st.button("💾 Save Configuration",on_click=reload_search_engine)
            
            

           
def Relevant_Information(relevant_list):
    """Display relevant information for a specific message or the latest one"""
    with st.expander("Relevant Information"):
       display_retrieval_list(relevant_list)
       
       
def show_confirmation_dialog(message):
    """Show a confirmation dialog"""
    dialog = st.empty()
    with dialog.container():
        st.write(message)
        col1, col2 = st.columns([1, 1])
        
        if col1.button("yes", key="confirm"):
            dialog.empty()  
            return True
        if col2.button("no", key="cancel"):
            dialog.empty() 
            return False
        
def reload_search_engine():
    """Reload the search engine with current config"""
    save_config(st.session_state.web_ui_config_path)
    if st.session_state.settings.get('engine_running'):
        st.session_state.settings['search_engine'] = NGSearch(NGConfig(all_config()))
        if st.session_state.main_folder != st.session_state.config['main_folder']:
            st.session_state.main_folder = st.session_state.config['main_folder']
            st.session_state.indices = json.load(open(os.path.join(st.session_state.config['main_folder'], 'info/indices.json'), 'r'))
            st.session_state.original_config_path = os.path.join(st.session_state.config['main_folder'], 'Node_config.yaml')
            st.session_state.web_ui_config_path = os.path.join(st.session_state.config['main_folder'], 'web_ui_config.yaml')
        return True
    return False


# Main chat interface
if os.path.exists(st.session_state.web_ui_config_path):
    load_config(st.session_state.web_ui_config_path)
elif os.path.exists(st.session_state.original_config_path):
    load_config(st.session_state.original_config_path)
else:
    NGConfig.create_config_file(main_folder)
    load_config(st.session_state.original_config_path)
display_header()
sidebar()
display_chat_history()
handle_user_input()
