import random
import re

def check_balance(message):
    balance = random.randint(500, 10000)
    return f"Your current account balance is ${balance:,.2f}."

def get_last_transaction(message):
    transactions = [
        "-$25.50 at 'Starbucks'",
        "-$112.00 at 'Amazon'",
        "+$1,500.00 'Paycheck Deposit'"
    ]
    last_tx = random.choice(transactions)
    return f"Your last transaction was: {last_tx}."

def transfer_funds(message):
    amount = None
    recipient = None

    amount_match = re.search(r'\$?([0-9,]+(\.[0-9]{2})?)', message)
    if amount_match:
        amount = amount_match.group(1)

    recipient_match = re.search(r'to\s+([A-Za-z]+)', message, re.IGNORECASE)
    if recipient_match:
        recipient = recipient_match.group(1).capitalize()

    if amount and recipient:
        return f"Confirming transfer of ${amount} to {recipient}..."
    elif amount:
        return f"You mentioned ${amount}, but who should I send it to?"
    elif recipient:
        return f"You mentioned {recipient}, but how much should I send?"
    else:
        return "To transfer funds, please specify an amount and a recipient."