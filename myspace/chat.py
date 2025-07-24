"""
Interactive chat interface for Agent TableRAG
Run this to have a conversation with your table-powered AI agent
"""
import sys
import os
from agent_tablerag import AgentTableRAG
import pandas as pd


class TableRAGChat:
    """Interactive chat interface for Agent TableRAG"""
    
    def __init__(self):
        self.agent = None
        self.setup_agent()
    
    def setup_agent(self):
        """Setup the agent with your data and explanation"""
        
        # ============= CUSTOMIZE THIS SECTION =============
        
        # Agent explanation - describe what your agent should do
        agent_explanation = """
        You are a personal meeting assistant that helps answer questions about my agenda.
        You should be helpful, accurate, and provide specific information from the tables when available.
        If you don't find relevant information, clearly state that.
        """
        
        # Table data configurations - add your own data here
        table_configs = [
            {
                "data_path": r"C:\Users\Bora\Desktop\RetrievalEnhancement\dataset\agenda_aug_dec_2025.xlsx",
                "table_name": "Aug-Dec 2025 Agenda",
                "explanation": "Meeting agenda for August to December 2025. Use this to answer questions about scheduled meetings, times, attendees, and locations."
            },
            # Add more tables as needed:
            # {
            #     "data_path": "path/to/your/data2.csv",
            #     "table_name": "YourTable2", 
            #     "explanation": "Description of second table"
            # }
        ]
        
        # ============= END CUSTOMIZATION =============
        
        print("🚀 Initializing Agent TableRAG...")
        
        try:
            # Initialize the agent
            self.agent = AgentTableRAG(
                agent_explanation=agent_explanation,
                config_path="config.json"
            )
            
            # Load tables
            for config in table_configs:
                data_path = config["data_path"]
                
                # Skip placeholder paths
                if data_path.startswith("path/to/your"):
                    print(f"⚠️  Skipping placeholder path: {data_path}")
                    print(f"   Please update the data_path in chat.py with your actual file path")
                    continue
                
                if not os.path.exists(data_path):
                    print(f"❌ File not found: {data_path}")
                    continue
                
                print(f"📊 Loading table: {config['table_name']}")

                df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
                
                result = self.agent.add_table_knowledge(
                    table_data=df,
                    knowledge_explanation=config["explanation"],
                    table_name=config["table_name"]
                )
                
                if result["status"] == "success":
                    print(f"✅ Successfully loaded {config['table_name']} ({result['num_chunks']} chunks)")
                else:
                    print(f"❌ Failed to load {config['table_name']}: {result.get('error', 'Unknown error')}")
            
            # Show summary
            summary = self.agent.get_table_summary()
            if summary["num_tables"] > 0:
                print(f"\n📋 Agent ready! Loaded {summary['num_tables']} tables with {summary['total_chunks']} total chunks")
                for table in summary['tables']:
                    print(f"   • {table['table_name']}: {table['num_chunks']} chunks")
            else:
                print("\n⚠️  No tables loaded. Please update the data paths in chat.py")
                print("    You can still test the system, but the agent won't have any table knowledge.")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            print("Please check your config.json and ensure all dependencies are installed.")
            sys.exit(1)
    
    def print_help(self):
        """Print help information"""
        print("\n💡 Available commands:")
        print("  • Type your question normally to chat with the agent")
        print("  • /help - Show this help message")
        print("  • /summary - Show agent and table summary")
        print("  • /confidence - Toggle confidence display on/off")
        print("  • /sources - Toggle source information on/off") 
        print("  • /clear - Clear conversation history")
        print("  • /quit or /exit - Exit the chat")
        print()
    
    def show_summary(self):
        """Show agent and table summary"""
        summary = self.agent.get_table_summary()
        
        print(f"\n📊 Agent Summary:")
        print(f"   Agent: {summary.get('agent_explanation', 'No explanation')}")
        print(f"   Tables: {summary['num_tables']}")
        print(f"   Total chunks: {summary['total_chunks']}")
        
        if summary['tables']:
            print("   📋 Loaded tables:")
            for table in summary['tables']:
                print(f"      • {table['table_name']}: {table['num_chunks']} chunks")
        print()
    
    def chat(self):
        """Main chat loop"""
        print("\n" + "="*60)
        print("🤖 Welcome to Agent TableRAG Chat!")
        print("Type your questions and the agent will answer using table data")
        print("Type /help for commands or /quit to exit")
        print("="*60)
        
        # Chat settings
        show_confidence = True
        show_sources = True
        conversation_count = 0
        
        self.print_help()
        
        while True:
            try:
                # Get user input
                print("👤 You: ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower().startswith('/'):
                    command = user_input.lower()
                    
                    if command in ['/quit', '/exit']:
                        print("👋 Goodbye!")
                        break
                    
                    elif command == '/help':
                        self.print_help()
                        continue
                    
                    elif command == '/summary':
                        self.show_summary()
                        continue
                    
                    elif command == '/confidence':
                        show_confidence = not show_confidence
                        print(f"🔧 Confidence display: {'ON' if show_confidence else 'OFF'}")
                        continue
                    
                    elif command == '/sources':
                        show_sources = not show_sources
                        print(f"🔧 Source information: {'ON' if show_sources else 'OFF'}")
                        continue
                    
                    elif command == '/clear':
                        conversation_count = 0
                        print("🧹 Conversation cleared")
                        continue
                    
                    else:
                        print(f"❓ Unknown command: {command}")
                        print("   Type /help to see available commands")
                        continue
                
                # Process the question
                print("🤖 Agent: ", end="")
                
                try:
                    response = self.agent.query(
                        user_question=user_input,
                        include_sources=show_sources,
                        highlight_matches=True
                    )
                    
                    # Print the answer
                    print(response["answer"])
                    
                    # Show confidence if enabled
                    if show_confidence:
                        confidence = response["confidence"]
                        confidence_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                        print(f"   {confidence_emoji} Confidence: {confidence:.2f}")
                    
                    # Show sources if enabled and available
                    if show_sources and response.get("sources"):
                        print(f"   📚 Sources ({response['num_sources']}):")
                        for i, source in enumerate(response["sources"][:3], 1):  # Show top 3
                            score = source["relevance_score"]
                            table_name = source["table_name"]
                            print(f"      {i}. {table_name} (relevance: {score:.3f})")
                    
                    conversation_count += 1
                    
                except Exception as e:
                    print(f"❌ Error processing your question: {e}")
                    print("   Please try rephrasing your question or check your configuration.")
                
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Continuing...")


def main():
    """Main function"""
    try:
        chat_interface = TableRAGChat()
        chat_interface.chat()
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure config.json has your OpenAI API key")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Update data paths in the chat.py file")


if __name__ == "__main__":
    main()
