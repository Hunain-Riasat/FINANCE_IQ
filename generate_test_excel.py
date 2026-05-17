"""
Simple script to generate the Excel test file from CSV data
Run this once to create test_transactions.xlsx
"""

import pandas as pd
import os

# Get absolute paths
script_dir = '/vercel/share/v0-project'
csv_file = os.path.join(script_dir, 'test_transactions.csv')
xlsx_file = os.path.join(script_dir, 'test_transactions.xlsx')

try:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create Excel file with formatting
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Transactions')
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Transactions']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print("[v0] Excel file created successfully!")
    print(f"[v0] Location: {xlsx_file}")
    print(f"[v0] Total transactions: {len(df)}")
    print(f"[v0] Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"[v0] Total income: ${df[df['Amount'] > 0]['Amount'].sum():,.2f}")
    print(f"[v0] Total expenses: ${abs(df[df['Amount'] < 0]['Amount'].sum()):,.2f}")
    
except Exception as e:
    print(f"[v0] Error creating Excel file: {str(e)}")
