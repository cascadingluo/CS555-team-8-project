// // Chatbot.ui.test.jsx
// import React from 'react';
// import { render, fireEvent, screen } from '@testing-library/react';
// import '@testing-library/jest-dom';
// import Chatbot from './src/components/Chatbot';
// import Login from './src/components/Login';
// import { BrowserRouter } from 'react-router-dom';

// const renderWithRouter = (component) => {
//   return render(<BrowserRouter>{component}</BrowserRouter>);
// };



// describe('Chatbot Component UI Tests', () => {
//   test('renders the chat button with the correct initial text', () => {
//     render(<Chatbot />);
    
//     // Verify the chat button is present with text "Chat"
//     const chatButton = screen.getByText('Chat');
//     expect(chatButton).toBeInTheDocument();
//   });

//   test('toggles chat window when button is clicked', () => {
//     render(<Chatbot />);
    
//     const chatButton = screen.getByText('Chat');

//     // Click to open the chat window
//     fireEvent.click(chatButton);
//     expect(screen.getByPlaceholderText('Type a message...')).toBeInTheDocument();
    
//     // Click again to close the chat window
//     fireEvent.click(chatButton);
//     expect(screen.queryByPlaceholderText('Type a message...')).not.toBeInTheDocument();
//   });

//   test('input field is empty by default and updates correctly', () => {
//     render(<Chatbot />);
    
//     // Open the chat window
//     fireEvent.click(screen.getByText('Chat'));
    
//     const inputField = screen.getByPlaceholderText('Type a message...');
    
//     // Verify the input field is initially empty
//     expect(inputField.value).toBe('');

//     // Change the input value
//     fireEvent.change(inputField, { target: { value: 'Hello' } });
//     expect(inputField.value).toBe('Hello');
//   });

//   test('send button is present and clickable', () => {
//     render(<Chatbot />);
    
//     // Open the chat window
//     fireEvent.click(screen.getByText('Chat'));
    
//     // Check if the Send button is present
//     const sendButton = screen.getByText('Send');
//     expect(sendButton).toBeInTheDocument();
    
//     // Click the send button (does not verify behavior, just checks interaction)
//     fireEvent.click(sendButton);
//   });

//   test('submitting a message clears the input field', () => {
//     render(<Chatbot />);

//     // Open the chat window
//     fireEvent.click(screen.getByText('Chat'));

//     const inputField = screen.getByPlaceholderText('Type a message...');
//     const sendButton = screen.getByText('Send');

//     // Change the input field value
//     fireEvent.change(inputField, { target: { value: 'Hello' } });

//     // Click the send button
//     fireEvent.click(sendButton);

//     // Verify the input field is cleared after submitting a message
//     expect(inputField.value).toBe('');
//   });
  
//   test('send button is enabled when input has text', () => {
//     render(<Chatbot />);
    
//     // Open the chat window
//     fireEvent.click(screen.getByText('Chat'));

//     const inputField = screen.getByPlaceholderText('Type a message...');
//     const sendButton = screen.getByText('Send');

//     // Change the input field value
//     fireEvent.change(inputField, { target: { value: 'Hello' } });

//     // Verify the send button is enabled when input is not empty
//     expect(sendButton).not.toBeDisabled();
//   });

//   test('renders login form correctly', () => {
//     renderWithRouter(<Login />);
    
//     // Check if username, password fields and submit button are rendered
//     expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
//     expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
//     expect(screen.getByRole('button', { name: /submit/i })).toBeInTheDocument();
//   });

//   test('allows user to type username and password', () => {
//     renderWithRouter(<Login />);

//     const usernameInput = screen.getByLabelText(/username/i);
//     const passwordInput = screen.getByLabelText(/password/i);

//     // Simulate typing
//     fireEvent.change(usernameInput, { target: { value: 'testUser' } });
//     fireEvent.change(passwordInput, { target: { value: 'testPassword' } });

//     // Check if input values are updated
//     expect(usernameInput.value).toBe('testUser');
//     expect(passwordInput.value).toBe('testPassword');
//   });

// });
