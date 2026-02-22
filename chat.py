from chatbot_assistant import ChatbotAssistant
from banking_functions import check_balance, transfer_funds, get_last_transaction

INTENTS_FILE = 'banking_intents.json'

if __name__ == '__main__':
    function_mappings = {
        'check_balance': check_balance,
        'transfer_funds': transfer_funds,
        'last_transaction': get_last_transaction
    }
    
    assistant = ChatbotAssistant(INTENTS_FILE)
    
    try:
        assistant.load_artifacts()
    except FileNotFoundError:
        print("Error: Model files not found.")
        print("Please run 'train.py' first to train the model.")
        exit()

    print("\nBanking Bot is ready! Type '/quit' to exit.")
    
    while True:
        message = input('You: ')
        if message == '/quit':
            break
        
        response = assistant.process_message(message, function_mappings)
        print(f"Bot: {response}")