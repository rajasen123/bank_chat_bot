from chatbot_assistant import ChatbotAssistant
import nltk

INTENTS_FILE = 'banking_intents.json'
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 200

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('wordnet')

    print("Starting training...")
    
    assistant = ChatbotAssistant(INTENTS_FILE)
    
    print("Parsing intents...")
    assistant.parse_intents()
    
    print("Preparing data...")
    assistant.prepare_data()
    
    print(f"Training model for {EPOCHS} epochs...")
    assistant.train_model(batch_size=BATCH_SIZE, lr=LEARNING_RATE, epochs=EPOCHS)
    
    print("Saving artifacts...")
    assistant.save_artifacts()
    
    print("\nTraining complete. Artifacts 'chatbot_model.pth' and 'training_data.json' saved.")