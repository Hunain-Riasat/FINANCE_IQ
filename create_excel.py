"""
Enhanced sample data generator with 365 days of realistic transaction data
Includes: income, expenses, payment methods, card types, and recurring payments
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_transactions():
    """Generate 365 days of realistic transaction data with all fields."""
    np.random.seed(42)
    
    transactions = []
    start_date = datetime(2024, 3, 6) - timedelta(days=365)
    
    # Merchant configuration
    merchants_config = {
        'Groceries': {
            'names': ['Whole Foods', 'Kroger', 'Safeway', "Trader Joe's", 'Costco', 'Sprouts', 'Aldi', 'Walmart'],
            'range': (30, 120),
            'frequency': 0.35
        },
        'Food & Dining': {
            'names': ['Chipotle', 'Starbucks', "McDonald's", 'Thai Restaurant', 'Sushi Place', 'Pizza Hut', 'Subway', 'KFC'],
            'range': (8, 45),
            'frequency': 0.45
        },
        'Utilities': {
            'names': ['Electric Company', 'Water Company', 'Internet Provider', 'Phone Company'],
            'range': (80, 180),
            'frequency': 1.0,
            'monthly': True
        },
        'Rent': {
            'names': ['Apartment Management', 'Landlord Payment', 'Property Management'],
            'range': (1200, 2500),
            'frequency': 1.0,
            'monthly': True
        },
        'Transportation': {
            'names': ['Shell Gas', 'Chevron', 'Uber', 'Lyft', 'Parking Garage', 'Public Transit'],
            'range': (3, 80),
            'frequency': 0.5
        },
        'Entertainment': {
            'names': ['Netflix', 'Spotify', 'Hulu', 'Disney+', 'Cinema', 'Concert Tickets'],
            'range': (15, 75),
            'frequency': 0.3
        },
        'Shopping': {
            'names': ['Amazon', 'Walmart', 'Target', 'H&M', 'Best Buy', 'Nike'],
            'range': (30, 150),
            'frequency': 0.15
        },
        'Healthcare': {
            'names': ['CVS Pharmacy', 'Walgreens', "Doctor's Office", 'Dental Clinic'],
            'range': (20, 500),
            'frequency': 0.02
        },
        'Subscriptions': {
            'names': ['Netflix', 'Spotify', 'Adobe Creative', 'Microsoft 365', 'Gym Membership'],
            'range': (9, 60),
            'frequency': 1.0,
            'monthly': True
        },
        'Income': {
            'names': ['Employer Salary', 'Freelance Project', 'Bonus'],
            'range': (3500, 4500),
            'frequency': 1.0,
            'monthly': True,
            'is_income': True
        }
    }
    
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer', 'Cash']
    card_types = ['Visa', 'Mastercard', 'American Express', 'Discover', 'Debit', 'Bank Transfer']
    
    # Generate transactions for 365 days
    for day_offset in range(365):
        current_date = start_date + timedelta(days=day_offset)
        
        for category, config in merchants_config.items():
            # Monthly transactions
            if config.get('monthly') and current_date.day == 1:
                merchant = np.random.choice(config['names'])
                amount = np.random.uniform(*config['range'])
                
                transactions.append({
                    'date': current_date,
                    'merchant_name': merchant,
                    'category': category,
                    'amount': -amount if config.get('is_income') else amount,
                    'description': f'{category} - {merchant}',
                    'payment_method': 'Bank Transfer' if config.get('is_income') else np.random.choice(payment_methods),
                    'card_type': 'Bank Transfer' if config.get('is_income') else np.random.choice(card_types),
                    'transaction_type': 'income' if config.get('is_income') else 'expense'
                })
            
            # Non-monthly transactions based on frequency
            elif not config.get('monthly') and np.random.random() < config['frequency']:
                merchant = np.random.choice(config['names'])
                amount = np.random.uniform(*config['range'])
                
                transactions.append({
                    'date': current_date + timedelta(hours=np.random.randint(6, 23)),
                    'merchant_name': merchant,
                    'category': category,
                    'amount': amount,
                    'description': f'{category} - {merchant}',
                    'payment_method': np.random.choice(payment_methods),
                    'card_type': np.random.choice(card_types),
                    'transaction_type': 'expense'
                })
    
    # Add some anomalies (fraud-like transactions)
    for _ in range(25):
        random_date = start_date + timedelta(days=np.random.randint(0, 365))
        transactions.append({
            'date': random_date,
            'merchant_name': f'Unknown Store {np.random.randint(100, 999)}',
            'category': np.random.choice(['Shopping', 'Entertainment', 'Utilities']),
            'amount': np.random.uniform(500, 2000),
            'description': 'Suspicious Transaction',
            'payment_method': 'Credit Card',
            'card_type': np.random.choice(card_types),
            'transaction_type': 'expense'
        })
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['transaction_id'] = [f'TXN{i:06d}' for i in range(len(df))]
    df['amount'] = df['amount'].round(2)
    
    # Reorder columns
    columns = ['transaction_id', 'date', 'merchant_name', 'category', 'amount', 
               'description', 'payment_method', 'card_type', 'transaction_type']
    df = df[columns]
    
    return df

# Generate data
df = generate_realistic_transactions()

# Save to CSV (relative path)
csv_path = 'sample_transactions.csv'
df.to_csv(csv_path, index=False)

# Save to Excel
try:
    xlsx_path = 'sample_transactions.xlsx'
    df.to_excel(xlsx_path, index=False, sheet_name='Transactions')
    print(f"[v0] Excel file created: {xlsx_path}")
except Exception as e:
    print(f"[v0] Excel creation skipped: {e}")

print(f"[v0] Sample data generation complete!")
print(f"[v0] Total transactions: {len(df)}")
print(f"[v0] Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"[v0] Total Income: ${abs(df[df['amount'] < 0]['amount'].sum()):,.2f}")
print(f"[v0] Total Expenses: ${df[df['amount'] > 0]['amount'].sum():,.2f}")
print(f"[v0] Net Balance: ${abs(df[df['amount'] < 0]['amount'].sum()) - df[df['amount'] > 0]['amount'].sum():,.2f}")
print(f"[v0] File saved: {csv_path}")
