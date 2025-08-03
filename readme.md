# NutriGenius - FastAPI Backend & ML Model

This repository contains the backend server and the machine learning model for NutriGenius, a smart recipe recommendation system. The backend is built with **Python** and **FastAPI**, and it serves personalized meal recommendations based on user-provided health data and preferences.

## ✨ Features

- **FastAPI Framework:** A modern, high-performance web framework for building APIs with Python.
- **Content-Based Filtering Model:** A custom-trained machine learning model that recommends recipes based on ingredient similarity, diet, course, and user goals.
- **Dynamic Image Fetching:** Uses DuckDuckGo Search to dynamically find relevant images for each recommended recipe, without needing API keys.
- **CORS Enabled:** Properly configured to accept requests from any frontend origin.
- **Ready for Deployment:** Includes a `Procfile` and `requirements.txt` for easy deployment on platforms like Render.

## 🛠️ Tech Stack

- **Language:** [Python](https://www.python.org/)
- **Web Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **ML & Data Processing:** [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/)
- **Image Search:** [duckduckgo-search](https://pypi.org/project/duckduckgo-search/)
- **Server:** [Uvicorn](https://www.uvicorn.org/)

## 🚀 Getting Started

Follow these instructions to get the backend server running on your local machine.

### Prerequisites

- Python (v3.9 or later)
- `pip` and `venv`

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/nutrigenius-backend.git](https://github.com/your-username/nutrigenius-backend.git)
    cd nutrigenius-backend
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Server

With your virtual environment activated, run the following command to start the local development server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation (powered by Swagger UI) at `http://127.0.0.1:8000/docs`.

## 📖 API Endpoints

### 1. Recommend Recipes

- **Endpoint:** `POST /recommend`
- **Description:** Takes user data and returns a list of personalized recipe recommendations.
- **Request Body:**
  ```json
  {
    "height_cm": 175,
    "weight_kg": 70,
    "desired_ingredients": "paneer, onion",
    "user_allergies": "nuts",
    "meal_type": "Dinner",
    "weight_goal": "Lose",
    "max_cooking_time": 60,
    "diet_preference": "Vegetarian"
  }
  ```
- **Success Response (200 OK):** An array of recipe objects.

### 2. Get Recipe Procedure

- **Endpoint:** `GET /procedure/{recipe_id}`
- **Description:** Retrieves the step-by-step cooking instructions for a specific recipe ID.
- **Success Response (200 OK):**
  ```json
  {
    "steps": ["Step 1...", "Step 2...", "Step 3..."]
  }
  ```

## 🚢 Deployment

This project is configured for easy deployment on [Render](https://render.com/).

1.  **Create `requirements.txt`:** If you've added new packages, update the file:

    ```bash
    pip freeze > requirements.txt
    ```

2.  **Create `Procfile`:** Ensure a `Procfile` exists in the root directory with the following content:

    ```
    web: uvicorn main:app --host 0.0.0.0 --port $PORT
    ```

3.  **Push to GitHub:** Commit and push your latest changes to a GitHub repository.

4.  **Deploy on Render:**
    - Create a new **Web Service** on Render and connect your GitHub repository.
    - Set the **Runtime** to `Python 3`.
    - Set the **Build Command** to `pip install -r requirements.txt`.
    - Set the **Start Command** to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
    - Add an environment variable with `PYTHON_VERSION` as the key and `3.11.9` as the value.
    - Click **Create Web Service**. Render will build and deploy your API.
