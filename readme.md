# NutriGenius Backend Setup Guide

Follow these steps to set up and run the server:

1. **Navigate to the backend directory:**

   ```
   cd C:\YourDirectory\NutriGenius\backend
   ```

2. **Create a virtual environment:**

   ```
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   ```
   venv\Scripts\activate
   ```

4. **Install the required dependencies:**

   ```
   pip install fastapi uvicorn pydantic httpx pandas scikit-learn numpy ddgs
   ```

5. **Start the FastAPI server:**
   ```
   uvicorn main:app --reload
   ```

The server will now be running locally. You can access the API documentation at `http://127.0.0.1:8000/docs`.
