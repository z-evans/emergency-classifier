# Application to create synthetic messages for disaster emergency messages

import json
import random
import csv
import sys


categories = json.load(open('./data/categories.json', 'r'))
disaster_structure = json.load(open('./data/disaster_sentence_structure.json', 'r'))
medical_structure = json.load(open('./data/medical_sentence_structure.json', 'r'))

# Get medical categories from categories.json
medical_category = next(cat for cat in categories if cat['id'] == 'medical')
medical_subcategories = [sub['id'] for sub in medical_category['sub']]

# Get disaster categories (all non-medical categories)
disaster_categories = [cat for cat in categories if cat['id'] != 'medical']


def generate_medical_message():
    """Generate a synthetic medical emergency message"""
    message_parts = []
    
    # Opening
    message_parts.append(random.choice(medical_structure['opening']))
    
    # Medical issue - pick subcategory
    subcategory = random.choice(medical_subcategories)
    issue = random.choice(medical_structure['categories'][medical_subcategories.index(subcategory)][subcategory])
    message_parts.append(issue)
    
    # Get priority for this subcategory
    priority = None
    for sub in medical_category['sub']:
        if sub['id'] == subcategory:
            priority = sub.get('priority', None)
            break
    
    # Optional: Add patient details (60% chance)
    if random.random() < 0.6:
        patient = random.choice(medical_structure['patient_details'])
        message_parts.append(f"Patient is {patient}")
    
    # Optional: Add consciousness level (40% chance)
    if random.random() < 0.4:
        consciousness = random.choice(medical_structure['consciousness_level'])
        message_parts.append(consciousness)
    
    # Optional: Add breathing status (40% chance)
    if random.random() < 0.4:
        breathing = random.choice(medical_structure['breathing_status'])
        message_parts.append(breathing)
    
    # Urgency (60% chance)
    if random.random() < 0.6:
        urgency_list = [u for u in medical_structure['urgency'] if u]  # Filter empty strings
        urgency = random.choice(urgency_list)
        message_parts.append(urgency)
    
    # Closing
    closing_list = [c for c in medical_structure['closing'] if c]  # Filter empty strings
    message_parts.append(random.choice(closing_list))
    
    message = '. '.join(message_parts) + '.'
    
    return {
        'category': 'medical',
        'subcategory': subcategory,
        'priority': priority,
        'message': message
    }


def generate_disaster_message():
    """Generate a synthetic disaster emergency message"""
    message_parts = []
    
    # Opening
    message_parts.append(random.choice(disaster_structure['opening']))
    
    # Pick a random disaster category
    category = random.choice(disaster_categories)
    category_id = category['id']
    
    # Find the category in disaster_structure
    disaster_category_data = None
    subcategory_id = None
    for cat_dict in disaster_structure['categories']:
        if category_id in cat_dict:
            disaster_category_data = cat_dict[category_id]
            break
    
    if disaster_category_data:
        # Pick a random subcategory
        subcategory_id = random.choice(list(disaster_category_data.keys()))
        disaster_phrase = random.choice(disaster_category_data[subcategory_id])
        message_parts.append(disaster_phrase)
        
        # Optional: Add additional context (40% chance)
        if random.random() < 0.4 and len(disaster_category_data[subcategory_id]) > 1:
            # Pick a different phrase from the same subcategory
            additional_phrases = [p for p in disaster_category_data[subcategory_id] if p != disaster_phrase]
            if additional_phrases:
                message_parts.append(random.choice(additional_phrases))
    
    # Get priority for this subcategory
    priority = None
    for sub in category['sub']:
        if sub['id'] == subcategory_id:
            priority = sub.get('priority', None)
            break
    
    # Closing
    message_parts.append(random.choice(disaster_structure['closing']))
    
    message = '. '.join(message_parts) + '.'
    
    return {
        'category': category_id,
        'subcategory': subcategory_id,
        'priority': priority,
        'message': message
    }


def generate_message(message_type=None):
    """
    Generate a synthetic emergency message.
    
    Args:
        message_type: 'medical', 'disaster', or None (random)
    
    Returns:
        dict with 'type', 'category', 'subcategory', 'priority', and 'message' keys
    """
    if message_type is None:
        message_type = random.choice(['medical', 'disaster'])
    
    if message_type == 'medical':
        result = generate_medical_message()
    else:
        result = generate_disaster_message()
    
    result['type'] = message_type
    return result


def generate_dataset(num_medical=100, num_disaster=100, output_file='synthetic_messages.csv'):
    """
    Generate a dataset of synthetic emergency messages.
    
    Args:
        num_medical: Number of medical messages to generate
        num_disaster: Number of disaster messages to generate
        output_file: Output CSV filename
    """
    messages = []
    
    # Generate medical messages
    print(f"Generating {num_medical} medical messages...")
    for _ in range(num_medical):
        msg = generate_message('medical')
        messages.append(msg)
    
    # Generate disaster messages
    print(f"Generating {num_disaster} disaster messages...")
    for _ in range(num_disaster):
        msg = generate_message('disaster')
        messages.append(msg)
    
    # Shuffle messages
    random.shuffle(messages)
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['type', 'category', 'subcategory', 'priority', 'message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for msg in messages:
            writer.writerow(msg)
    
    print(f"\nGenerated {len(messages)} messages and saved to {output_file}")
    print(f"Medical: {num_medical}, Disaster: {num_disaster}")


# Example usage
if __name__ == "__main__":
    # Generate a dataset if requested
    if len(sys.argv) > 1:
        # Validate number of arguments
        if len(sys.argv) < 4:
            print("Error: Insufficient arguments.")
            print("Usage: python main.py <num_medical> <num_disaster> <output_file>")
            print("Example: python main.py 100 100 messages.csv")
            sys.exit(1)
        
        # Parse and validate arguments
        try:
            num_medical = int(sys.argv[1])
            num_disaster = int(sys.argv[2])
            output_file = sys.argv[3]
            
            # Validate num_medical
            if num_medical <= 0:
                print(f"Error: num_medical must be a positive integer, got: {num_medical}")
                sys.exit(1)
            
            # Validate num_disaster
            if num_disaster <= 0:
                print(f"Error: num_disaster must be a positive integer, got: {num_disaster}")
                sys.exit(1)
            
            # Validate output file extension
            if not output_file.endswith('.csv'):
                print(f"Error: output_file must have .csv extension, got: {output_file}")
                sys.exit(1)
            
            # Warn if file exists
            import os
            if os.path.exists(output_file):
                print(f"Warning: {output_file} already exists and will be overwritten.")
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    print("Operation cancelled.")
                    sys.exit(0)
        
        except ValueError as e:
            print(f"Error: Invalid argument type. num_medical and num_disaster must be integers.")
            print(f"Details: {e}")
            sys.exit(1)
        
        generate_dataset(num_medical, num_disaster, output_file)
    else:
        # No arguments provided - generate and print sample messages
        print("=== Sample Medical Message ===")
        med_msg = generate_message('medical')
        print(f"Category: {med_msg['category']}, Subcategory: {med_msg['subcategory']}, Priority: {med_msg['priority']}")
        print(f"Message: {med_msg['message']}")
        
        print("\n=== Sample Disaster Message ===")
        dis_msg = generate_message('disaster')
        print(f"Category: {dis_msg['category']}, Subcategory: {dis_msg['subcategory']}, Priority: {dis_msg['priority']}")
        print(f"Message: {dis_msg['message']}")
        
        print("\n=== Random Message ===")
        msg = generate_message()
        print(f"Type: {msg['type']}, Category: {msg['category']}, Subcategory: {msg['subcategory']}, Priority: {msg['priority']}")
        print(f"Message: {msg['message']}")
        
        print("\n" + "="*60)
        print("To generate a dataset, use:")
        print("python main.py <num_medical> <num_disaster> <output_file>")
        print("Example: python main.py 100 100 messages.csv")
        print("="*60)