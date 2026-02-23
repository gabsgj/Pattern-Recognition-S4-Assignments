

---

# Baum-Welch Hidden Markov Model (HMM) Dashboard

**Name:** Rohan Shyam

**Roll Number:** TCR24CS056

---

## 🚀 Project Overview

An interactive dashboard for analyzing Hidden Markov Models using the Baum-Welch algorithm. This project allows users to input observation sequences and visualize the optimization of Transition (A) and Emission (B) matrices.

### 🛠️ Tech Stack

* **Backend:** Django, Django REST Framework, NumPy
* **Frontend:** Next.js, React, Tailwind CSS
* **Environment:** Developed and tested on Fedora Linux

---

## 💻 Local Execution Instructions



### 1. Prerequisites

Ensure you have **Python 3.11+** and **Node.js** installed on your machine.


```bash
git clone https://github.com/rohan-shyam/Baum-Welch-Algorithm/
cd Baum-Welch-Algorithm
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

```

3. Navigate to the server directory:
```bash
cd server

```


4. Run the server:
```bash
python manage.py runserver

```


*The API will be live at `http://127.0.0.1:8000*`

### 3. Frontend Setup

1. Open a new terminal.
2. Navigate to the client directory:
```bash
cd client

```


3. Install dependencies and run the development server:
```bash
npm install
npm run dev

```


*Access the dashboard at `http://localhost:3000*`

---

## ⚙️ Core Logic: Baum-Welch Algorithm

The algorithm performs iterative expectation-maximization to find the maximum likelihood estimates of HMM parameters.

1. **Expectation (E-step):** Calculates forward () and backward () probabilities.
2. **Maximization (M-step):** Updates the A, B, and  matrices based on the calculated probabilities.

---

### Screenshots
<img width="1223" height="855" alt="image" src="https://github.com/user-attachments/assets/8b8fe963-3d2b-4fc7-be7a-385c0fd73ae0" />

<img width="1547" height="922" alt="image" src="https://github.com/user-attachments/assets/b3b6f727-7ed0-4e94-80b1-802544ea97e4" />


