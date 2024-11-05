// backend/server.js
import dotenv from "dotenv";
import router from './routes/user.routes.js';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { join } from 'path';
import { connectDB } from './config/db.js';
import User from './models/user.model.js';

const app = express();
app.use(express.json());
app.use(cors());

app.use("/api/user", router);

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
//const model = genAI.getGenerativeModel({ model: "tunedModels/my-increment-model" });

const initialPrompts = ["Hey there! How are you feeling today?","Hi! I'm here for you. What’s on your mind?","Hello! How’s everything going for you today?","Hey! I’m ready to listen. How are you feeling?",
"Hi there! How are you doing, both physically and mentally?","Hello! What’s something you’d like to talk about today?","Hi! How’s your day been so far?","Hey! It’s good to see you. How are you holding up?",
"Hello! How can I support you today?","Hi! What’s been on your mind lately?","Hey! I’m here to help you feel better. How are you?","Hi! I’d love to hear how you’re doing today.",
"Hello! Is there anything on your mind that you'd like to share?","Hey! How’s your mood today?","Hi there! How are you feeling emotionally today?", "Hello! It’s great to see you. How’s everything going for you?",
"Hi! How are you feeling in this moment?", "Hey! I’m here if you need someone to talk to. How are you?","Hello! How’s your mental health been lately?","Hi! Let’s chat. How are you feeling today?",
"Hey! How’s your headspace today? Need a little support?","Hello! What’s something on your mind that we can talk about?","Hi there! How are things going for you today?","Hey! How are you really feeling today?",
"Hello! How has your week been so far?","Hi! Anything on your mind that you’d like to talk about?","Hey! How can I be helpful to you today?","Hi there! How’s your emotional well-being right now?",
"Hello! Let’s check in. How are you doing today?","Hi! What’s something you’ve been thinking about lately?","Hey! How have you been feeling emotionally this week?","Hello! How’s your mental health today?",
"Hi! I’m here for you. What’s been on your mind?","Hey! How are you feeling today, physically and mentally?","Hi there! Anything you’d like to talk about today?","Hello! How are you feeling emotionally and physically?",
"Hi! Let’s take a moment to check in. How are you feeling?","Hey! How’s your stress level today? Need to talk?","Hello! I’m ready to listen. How’s your day been?","Hi! How’s your mental space right now?",
"Hey! What’s one thing you’ve been thinking about lately?","Hello! How’s your energy level today? How are you feeling?","Hi! It’s great to check in with you. How are you feeling?","Hey! Let’s chat about how you’re doing today.",
"Hello! What’s been on your mind this week?","Hi! I’m here for you. How’s your mental health?","Hey! How’s your day going? Need someone to talk to?","Hello! How are you feeling emotionally and mentally?",
"Hi there! How can I help support you today?","Hey! What’s something you’ve been feeling lately?"
]; 

//cahtbot api init
app.post('/api/initChat', async (req, res) => {
  const initialPrompt = initialPrompts[Math.floor(Math.random() * initialPrompts.length)];
  
  try {
    const result = await model.generateContent(initialPrompt);
    const botResponse = await result.response.text(); // Ensure you await if necessary
    res.json({ reply: initialPrompt });
  } catch (error) {
    console.error("Error generating initial response:", error);
    res.status(500).json({ error: "Error generating initial response" });
  }
});

app.post('/api/chat', async (req, res) => {
  const userInput = req.body.input;
  const {userId} = req.body;

  const prompt = `You are a compassionate and knowledgeable mental health assistant. Your role is to listen to the user’s emotional concerns and 
    provide supportive advice. Act as a friend or caretaker to the user. Use simple, everyday language to keep things easy to understand. Make 
    sure your tone is conversational, and avoid technical terms or complex phrases. At the end ask a specific follow-up question related to their 
    health or well-being. Keep your ques;cdtions friendly and focused on keeping the conversation going in a positive, engaging way. Provide the best 
    course of action based on this input: ${userInput} Respond in 3-4 sentences.`;

    try {
      const result = await model.generateContent(prompt);
      const botResponse = await result.response.text(); // Ensure you await if necessary
      if (userId){
        try{
          await User.findByIdAndUpdate(userId, {
            $push: {
              chat_history: [
              {
                role: 'user',
                content: userInput,
                timestamp: new Date()
              }, 
              {
                role: 'bot',
                content: botResponse,
                timestamp: new Date()
              },
            ],
            },
          });
          console.log("Chat history saved successfully.");
        } catch(dbError) {
          console.error("Database error:", dbError);
        }
      }

      res.json({ reply: botResponse });
  } catch (error) {
    console.error("Error generating response:", error);
    res.status(500).json({ error: "Error generating response" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  connectDB();
  console.log("Server started at http://localhost:" + PORT);
});