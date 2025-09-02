import csv
import os



def add_contact():
    name = input("Enter your name: ")
    contact = input("Enter your contact number: ")
    print(f"Name: {name}, Contact: {contact}")
    # Check if the contacts.csv file exists
    file_exists = os.path.isfile('contacts.csv')
    # Open the file in append mode and write the contact information
    with open('contacts.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file does not exist
        if not file_exists:
            writer.writerow(['Name', 'Contact'])
            writer.writerow([name, contact])
        else:
            # Write the user's name and contact
            writer.writerow([name, contact])

def view_contacts():
    
    if not os.path.isfile('contacts.csv'):
        print("No contacts found.")
        return

    # Open the file in read mode and display the contacts
    with open('contacts.csv', mode='r', newline='') as file:
        reader = csv.reader(file)
        contacts = list(reader)
        # If only header exists or file is empty, show no contacts
        if len(contacts) <= 1:
            print("No contacts found.")
            return
        # Print each contact row
        for row in contacts:
            print('\t'.join(row))

def main():
    while True:
        print("\nMenu:")
        print("1. Add Contact")
        print("2. View Contacts")
        print("3. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            while True:
                add_contact()
                answer = input("Do you want to add another member? Answer yes or no: ").lower()
                if answer != "yes":
                    break
        elif choice == '2':
            view_contacts()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
