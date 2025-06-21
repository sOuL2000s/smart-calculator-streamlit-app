# Smart Calculator App - Pro Version (All-in-One Math, Stats, Finance, etc.)

import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import re
import math
from datetime import datetime, timedelta

# --- Solver Functions ---
def solve_addition(values):
    try:
        numbers = list(map(float, values.split()))
        return f"Sum: {sum(numbers)}"
    except:
        return "❌ Invalid input for addition."

def solve_subtraction(values):
    try:
        numbers = list(map(float, values.split()))
        result = numbers[0] - sum(numbers[1:])
        return f"Result: {result}"
    except:
        return "❌ Invalid input for subtraction."

def solve_multiplication(values):
    try:
        numbers = list(map(float, values.split()))
        result = np.prod(numbers)
        return f"Product: {result}"
    except:
        return "❌ Invalid input for multiplication."

def solve_division(values):
    try:
        numbers = list(map(float, values.split()))
        result = numbers[0]
        for n in numbers[1:]:
            result /= n
        return f"Quotient: {result}"
    except:
        return "❌ Invalid input for division."

def solve_trigonometry(expr):
    try:
        return str(sp.sympify(expr).evalf())
    except:
        return "❌ Invalid trigonometric expression."

def solve_equation(expr):
    try:
        x = sp.Symbol('x')
        eq = sp.Eq(*map(sp.sympify, expr.split('=')))
        return f"Solution: {sp.solve(eq, x)}"
    except:
        return "❌ Invalid algebraic equation."

def solve_calculus(expr):
    try:
        return f"Result: {sp.sympify(expr)}"
    except:
        return "❌ Invalid calculus expression."

def solve_statistics(expr):
    try:
        data = list(map(float, expr.split()))
        return f"Mean: {np.mean(data)}, Median: {np.median(data)}, Std Dev: {np.std(data)}"
    except:
        return "❌ Invalid statistics input."

def solve_probability(expr):
    try:
        total, event = map(float, expr.split())
        prob = event / total
        return f"Probability: {prob:.4f} or {prob*100:.2f}%"
    except:
        return "❌ Use: total_cases event_cases"

def solve_matrix(expr):
    try:
        mat = np.array(eval(expr))
        det = np.linalg.det(mat)
        inv = np.linalg.inv(mat) if np.linalg.det(mat) != 0 else "Not Invertible"
        return f"Determinant: {det}\nInverse:\n{inv}"
    except:
        return "❌ Invalid matrix format. Use [[1,2],[3,4]]"

def solve_unit_conversion(expr):
    try:
        value, from_unit, to_unit = expr.split()
        value = float(value)
        conversions = {
            ("km", "m"): 1000,
            ("m", "km"): 0.001,
            ("cm", "m"): 0.01,
            ("m", "cm"): 100,
            ("kg", "g"): 1000,
            ("g", "kg"): 0.001
        }
        return f"Converted: {value * conversions[(from_unit, to_unit)]} {to_unit}"
    except:
        return "❌ Format: value from_unit to_unit"

def solve_datetime(expr):
    try:
        date1, date2 = expr.split()
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        delta = abs((d2 - d1).days)
        return f"Difference: {delta} days"
    except:
        return "❌ Format: YYYY-MM-DD YYYY-MM-DD"

def solve_finance(expr):
    try:
        principal, rate, time = map(float, expr.split())
        si = (principal * rate * time) / 100
        amount = principal + si
        return f"Simple Interest: {si}\nTotal Amount: {amount}"
    except:
        return "❌ Format: principal rate time"

def solve_plot(expr):
    try:
        x = sp.Symbol('x')
        parsed = sp.sympify(expr)
        f = sp.lambdify(x, parsed, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        ax.set_title(f"Graph of {expr}")
        st.pyplot(fig)
        return "Graph rendered successfully."
    except:
        return "❌ Unable to plot graph."

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Smart Calculator Pro", layout="centered", page_icon="🧮")
st.title("🧠 Smart Calculator Pro")
st.caption("Every domain. One calculator. 💡")

operation = st.selectbox("📌 Select Operation:", [
    "Addition", "Subtraction", "Multiplication", "Division",
    "Trigonometry", "Algebraic Equation", "Calculus Expression",
    "Statistics Summary", "Probability (event/total)", "Matrix Ops", 
    "Unit Conversion", "Date Difference", "Simple Interest",
    "Graph Expression"
])

# --- Syntax Guidance ---
syntax_examples = {
    "Addition": "Example: 10 20 30",
    "Subtraction": "Example: 100 25 10",
    "Multiplication": "Example: 2 3 4",
    "Division": "Example: 100 5 2",
    "Trigonometry": "Example: sin(pi/4)",
    "Algebraic Equation": "Example: x**2 + 5*x + 6 = 0",
    "Calculus Expression": "Example: diff(x**2, x)",
    "Statistics Summary": "Example: 10 20 30 40",
    "Probability (event/total)": "Example: 100 25",
    "Matrix Ops": "Example: [[1,2],[3,4]]",
    "Unit Conversion": "Example: 1000 g kg",
    "Date Difference": "Example: 2025-01-01 2025-06-01",
    "Simple Interest": "Example: 10000 5 2",
    "Graph Expression": "Example: x**2 - 4*x + 3"
}

st.markdown(f"**🧾 Syntax:** {syntax_examples.get(operation, 'Enter valid input')}\n")

user_input = st.text_input("✍️ Enter input:")

if user_input:
    if operation == "Addition":
        result = solve_addition(user_input)
    elif operation == "Subtraction":
        result = solve_subtraction(user_input)
    elif operation == "Multiplication":
        result = solve_multiplication(user_input)
    elif operation == "Division":
        result = solve_division(user_input)
    elif operation == "Trigonometry":
        result = solve_trigonometry(user_input)
    elif operation == "Algebraic Equation":
        result = solve_equation(user_input)
    elif operation == "Calculus Expression":
        result = solve_calculus(user_input)
    elif operation == "Statistics Summary":
        result = solve_statistics(user_input)
    elif operation == "Probability (event/total)":
        result = solve_probability(user_input)
    elif operation == "Matrix Ops":
        result = solve_matrix(user_input)
    elif operation == "Unit Conversion":
        result = solve_unit_conversion(user_input)
    elif operation == "Date Difference":
        result = solve_datetime(user_input)
    elif operation == "Simple Interest":
        result = solve_finance(user_input)
    elif operation == "Graph Expression":
        result = solve_plot(user_input)
    else:
        result = "❌ Unknown operation."

    if isinstance(result, str):
        if "Graph" not in result:
            st.success(result)
    else:
        st.success("✅ Done")

st.markdown("""
---
💹 *Supported Formats:*
- Trig: `sin(pi/4)`
- Algebra: `x**2 + 5*x + 6 = 0`
- Calculus: `diff(x**2, x)` or `integrate(x**2, x)`
- Matrix: `[[1,2],[3,4]]`
- Units: `1000 g kg`
- Dates: `2025-01-01 2025-06-01`
- Finance: `10000 5 2` (₹, %, years)
---
Made with ❤️ using Streamlit + NumPy + SymPy + Matplotlib
""")