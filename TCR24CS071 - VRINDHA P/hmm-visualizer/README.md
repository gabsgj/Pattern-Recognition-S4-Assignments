# Pattern Recognition Assignment: HMM Baum-Welch Visualizer

**Name:** Vrindha P  
**University Register Number:** TCR24CS071 

## Description
This project is an interactive web application that visually implements the Hidden Markov Model (HMM) using the Baum-Welch algorithm. It is built using React and Vite. 

Instead of just outputting numbers to a console, this application provides a full graphical user interface to:
- Interactively edit the Initial Distribution ($\pi$), Transition Matrix ($A$), and Emission Matrix ($B$).
- Step through the Baum-Welch algorithm mathematically.
- Visualize state transitions using dynamic graphs.
- Plot optimization charts demonstrating how $P(O|\lambda)$ and $1 - P(O|\lambda)$ converge over iterations.

## Prerequisites
To run this application, you need to have the following installed on your computer:
- **Node.js** (v14 or higher)
- **npm** (Node Package Manager - comes with Node.js)

## Instructions to Run Your Program Locally

1. **Open your terminal** and navigate into the app folder (where `package.json` is located):
   ```bash
   cd hmm-visualizer
   ```

2. **Install the dependencies:**
   Run the following command to download all required React packages:
   ```bash
   npm install
   ```

3. **Start the development server:**
   Run the following command to boot up the application:
   ```bash
   npm run dev
   ```

4. **View the App:**
   Once the server starts, it will provide a local URL in your terminal (usually `http://localhost:5173/`). `CTRL` + Click (or `CMD` + Click on Mac) that link to open the visualizer in your web browser.
