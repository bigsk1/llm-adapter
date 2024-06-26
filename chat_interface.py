import sys
import logging
import os
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from llm_adapter import UniversalLLMAdapter
from dotenv import load_dotenv

load_dotenv()
console = Console()

def setup_logging():
    logging.basicConfig(
        level="WARNING",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True), logging.FileHandler("error.log")]
    )

class ChatInterface:
    def __init__(self):
        self.console = Console()

    def display_message(self, message, sender="System", end="\n"):
        if sender == "AI":
            self.console.print(message, end=end)
        else:
            panel = Panel(
                Markdown(message),
                title=sender,
                border_style="bold",
                padding=(1, 1)
            )
            self.console.print(panel)

    def get_user_input(self):
        user_input = Prompt.ask("You")
        self.console.print()  
        return user_input

    def display_model_info(self):
        text = Text()
        text.append("Selected Model: ", style="bold")
        text.append(f"{os.getenv('MODEL_NAME')}\n\n", style="green")
        text.append(f"Provider: {os.getenv('PROVIDER')}\n\n")
        text.append("Settings:\n", style="bold")
        text.append(f"Temperature: {os.getenv('TEMPERATURE')}\n")
        text.append(f"Top P: {os.getenv('TOP_P')}\n")
        text.append(f"Max Tokens: {os.getenv('MAX_TOKENS')}\n")
        
        panel = Panel(
            text,
            title="Model Information",
            border_style="bold",
            padding=(1, 1)
        )
        self.console.print(panel)

def main():
    setup_logging()
    
    if not os.getenv('MODEL_NAME') or not os.getenv('PROVIDER'):
        console.print("[bold red]Error: MODEL_NAME and PROVIDER must be set in the .env file.[/bold red]")
        sys.exit(1)

    llm_adapter = UniversalLLMAdapter()
    chat_interface = ChatInterface()

    console.print(f"[bold green]Connected to {os.getenv('MODEL_NAME')} model via {os.getenv('PROVIDER')}.[/bold green]")
    chat_interface.display_model_info()
    chat_interface.display_message(f"Welcome to the Universal LLM Chat Interface!", "System")

    while True:
        try:
            user_input = chat_interface.get_user_input()
            if user_input.lower() in ["exit", "quit", "bye"]:
                chat_interface.display_message("Goodbye!", "System")
                break
            for response_chunk in llm_adapter.send_request(user_input):
                chat_interface.display_message(response_chunk, "AI", end="")
            print()  # New line after the complete response
        except KeyboardInterrupt:
            chat_interface.display_message("Interrupted. Exiting...", "System")
            break
        except Exception as e:
            logging.exception("An error occurred:")
            chat_interface.display_message(f"Error: {str(e)}", "System")

if __name__ == "__main__":
    main()