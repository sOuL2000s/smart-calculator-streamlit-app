import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json # For safer parsing of matrix input

# --- Solver Functions ---

def solve_addition(values: str) -> str:
    """Calculates the sum of space-separated numbers."""
    try:
        numbers = list(map(float, values.split()))
        if not numbers:
            return "❌ No numbers provided for addition."
        return f"Sum: {sum(numbers):.4f}"
    except ValueError:
        return "❌ Invalid input for addition. Please enter space-separated numbers (e.g., '10 20 30')."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_subtraction(values: str) -> str:
    """Calculates the result of subtracting subsequent numbers from the first."""
    try:
        numbers = list(map(float, values.split()))
        if not numbers:
            return "❌ No numbers provided for subtraction."
        if len(numbers) == 1:
            return f"Result: {numbers[0]:.4f}" # If only one number, return itself
        result = numbers[0] - sum(numbers[1:])
        return f"Result: {result:.4f}"
    except ValueError:
        return "❌ Invalid input for subtraction. Please enter space-separated numbers (e.g., '100 25 10')."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_multiplication(values: str) -> str:
    """Calculates the product of space-separated numbers."""
    try:
        numbers = list(map(float, values.split()))
        if not numbers:
            return "❌ No numbers provided for multiplication."
        result = np.prod(numbers)
        return f"Product: {result:.4f}"
    except ValueError:
        return "❌ Invalid input for multiplication. Please enter space-separated numbers (e.g., '2 3 4')."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_division(values: str) -> str:
    """Calculates the quotient of dividing the first number by subsequent numbers."""
    try:
        numbers = list(map(float, values.split()))
        if len(numbers) < 2:
            return "❌ At least two numbers are required for division."
        result = numbers[0]
        for n in numbers[1:]:
            if n == 0:
                return "❌ Division by zero is not allowed."
            result /= n
        return f"Quotient: {result:.4f}"
    except ValueError:
        return "❌ Invalid input for division. Please enter space-separated numbers (e.g., '100 5 2')."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_trigonometry(expr: str) -> str:
    """Evaluates trigonometric expressions using SymPy. Use 'pi' for pi."""
    try:
        # SymPy requires variables to be declared, handle common constants like pi
        expr_parsed = sp.sympify(expr, locals={'pi': sp.pi, 'e': sp.E})
        result = expr_parsed.evalf()
        return f"Result: {result:.6f}" # Increased precision for trig
    except (sp.SympifyError, SyntaxError, TypeError):
        return "❌ Invalid trigonometric expression. Example: `sin(pi/4)`, `cos(0)`. Use `pi` for $\\pi$."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_equation(expr: str) -> str:
    """Solves algebraic equations for 'x' using SymPy."""
    try:
        x = sp.Symbol('x')
        if '=' not in expr:
            return "❌ Invalid algebraic equation. Missing `=`. Example: `x**2 + 5*x + 6 = 0`."
        lhs, rhs = expr.split('=')
        eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
        solutions = sp.solve(eq, x)
        if not solutions:
            return "✅ No real solutions found, or the equation is an identity/contradiction."
        return f"Solution: {solutions}"
    except (sp.SympifyError, SyntaxError, TypeError):
        return "❌ Invalid algebraic equation. Example: `x**2 + 5*x + 6 = 0`. Ensure variable is `x`."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_calculus(expr: str) -> str:
    """Evaluates calculus expressions (differentiation or integration) using SymPy.
    Examples: `diff(x**2, x)`, `integrate(x**3, x)`.
    """
    try:
        x = sp.Symbol('x') # Ensure x is defined for calculus operations
        # Sympy handles parsing 'diff(x**2, x)' or 'integrate(x**2, x)' directly
        result = sp.sympify(expr)
        return f"Result: {result}"
    except (sp.SympifyError, SyntaxError, TypeError):
        return "❌ Invalid calculus expression. Examples: `diff(x**2, x)`, `integrate(x**3, x)`."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_statistics(expr: str) -> str:
    """Calculates mean, median, and standard deviation for space-separated numbers."""
    try:
        data = list(map(float, expr.split()))
        if not data:
            return "❌ No data provided for statistics."
        if len(data) < 2:
            return "❌ At least two numbers are recommended for meaningful statistics (mean, median, std dev)."

        mean = np.mean(data)
        median = np.median(data)
        std_dev = np.std(data) # Population standard deviation

        # Add more stats
        data_series = pd.Series(data) # Using pandas for convenience for other stats
        variance = data_series.var(ddof=0) # Population variance
        data_range = np.max(data) - np.min(data)

        return (f"Mean: {mean:.4f}, Median: {median:.4f}, "
                f"Std Dev: {std_dev:.4f}, Variance: {variance:.4f}, "
                f"Range: {data_range:.4f}")
    except ValueError:
        return "❌ Invalid statistics input. Please enter space-separated numbers (e.g., '10 20 30 40')."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_probability(expr: str) -> str:
    """Calculates probability from total cases and event cases."""
    try:
        parts = expr.split()
        if len(parts) != 2:
            return "❌ Invalid probability format. Use: `total_cases event_cases`."
        total, event = map(float, parts)
        if total <= 0:
            return "❌ Total cases must be greater than zero."
        if event < 0:
            return "❌ Event cases cannot be negative."
        if event > total:
            return "❌ Event cases cannot exceed total cases."
        prob = event / total
        return f"Probability: {prob:.4f} or {prob*100:.2f}%"
    except ValueError:
        return "❌ Invalid probability input. Please ensure total_cases and event_cases are numbers."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_matrix(expr: str) -> str:
    """Performs matrix operations (determinant, inverse). Supports square matrices."""
    try:
        # Use json.loads for safer parsing of list of lists
        mat_list = json.loads(expr)
        mat = np.array(mat_list)

        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            return "❌ Invalid matrix format. Only square matrices are supported. Example: `[[1,2],[3,4]]`."
        if mat.size == 0:
            return "❌ Empty matrix provided."

        det = np.linalg.det(mat)
        det_str = f"{det:.6f}" # Format determinant for readability

        if abs(det) < 1e-9: # Check for near-zero determinant for invertibility
            inv = "Not Invertible (Determinant is close to zero or zero)."
        else:
            inv = np.linalg.inv(mat)
            inv = str(np.array2string(inv, separator=', ', formatter={'float_kind':lambda x: "%.6f" % x})) # Format inverse nicely

        return f"Determinant: {det_str}\nInverse:\n{inv}"
    except json.JSONDecodeError:
        return "❌ Invalid matrix format. Please use valid JSON-like list of lists (e.g., `[[1,2],[3,4]]`)."
    except ValueError as ve:
        return f"❌ Matrix error: {ve}. Ensure all rows have the same number of columns and contain valid numbers."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_unit_conversion(expr: str) -> str:
    """Converts values between specified units.
    Format: `value from_unit to_unit`
    """
    try:
        parts = expr.lower().split()
        if len(parts) != 3:
            return "❌ Invalid unit conversion format. Use: `value from_unit to_unit`."
        value = float(parts[0])
        from_unit = parts[1]
        to_unit = parts[2]

        conversions = {
            ("km", "m"): 1000, ("m", "km"): 0.001,
            ("cm", "m"): 0.01, ("m", "cm"): 100,
            ("kg", "g"): 1000, ("g", "kg"): 0.001,
            ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
            ("celsius", "fahrenheit"): lambda c: (c * 9/5) + 32,
            ("fahrenheit", "celsius"): lambda f: (f - 32) * 5/9,
            ("liter", "ml"): 1000, ("ml", "liter"): 0.001,
            ("hour", "minute"): 60, ("minute", "hour"): 1/60,
        }

        # Check for direct conversion
        if (from_unit, to_unit) in conversions:
            conversion_factor = conversions[(from_unit, to_unit)]
            if callable(conversion_factor): # Handle temperature conversions
                converted_value = conversion_factor(value)
            else:
                converted_value = value * conversion_factor
            return f"Converted: {converted_value:.4f} {to_unit}"
        else:
            return f"❌ Unsupported unit conversion: `{from_unit}` to `{to_unit}`. Try `km/m`, `kg/g`, `m/ft`, `celsius/fahrenheit`, `liter/ml`, `hour/minute`."
    except ValueError:
        return "❌ Invalid value for unit conversion. Ensure the value is a number."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_datetime(expr: str) -> str:
    """Calculates the difference in days between two dates.
    Format: `YYYY-MM-DD YYYY-MM-DD`
    """
    try:
        parts = expr.split()
        if len(parts) != 2:
            return "❌ Invalid date format. Use: `YYYY-MM-DD YYYY-MM-DD`."
        date1_str, date2_str = parts
        d1 = datetime.strptime(date1_str, "%Y-%m-%d")
        d2 = datetime.strptime(date2_str, "%Y-%m-%d")
        delta = abs((d2 - d1).days)
        return f"Difference: {delta} days"
    except ValueError:
        return "❌ Invalid date format. Please use `YYYY-MM-DD` (e.g., `2025-01-01 2025-06-01`)."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_finance(expr: str) -> str:
    """Calculates simple interest and total amount.
    Format: `principal rate time` (Rate in percentage, Time in years).
    """
    try:
        parts = expr.split()
        if len(parts) != 3:
            return "❌ Invalid finance format. Use: `principal rate time` (e.g., `10000 5 2`)."
        principal, rate, time = map(float, parts)
        if principal < 0:
            return "❌ Principal cannot be negative."
        if rate < 0:
            return "❌ Rate cannot be negative."
        if time < 0:
            return "❌ Time cannot be negative."

        si = (principal * rate * time) / 100
        amount = principal + si
        return f"Simple Interest: {si:.2f}\nTotal Amount: {amount:.2f}"
    except ValueError:
        return "❌ Invalid finance input. Ensure principal, rate, and time are numbers."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def solve_plot(expr: str):
    """Plots a mathematical expression involving 'x'."""
    try:
        x = sp.Symbol('x')
        parsed = sp.sympify(expr)
        f = sp.lambdify(x, parsed, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        ax.set_title(f"Graph of {expr}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True) # Add grid for better readability
        st.pyplot(fig)
        plt.close(fig) # Close the figure to prevent memory issues

        return "Graph rendered successfully."
    except (sp.SympifyError, TypeError, ValueError, NameError):
        return "❌ Unable to plot graph. Ensure it's a valid mathematical expression involving `x`."
    except Exception as e:
        return f"❌ An unexpected error occurred during plotting: {e}"


# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Smart Calculator Pro", layout="centered", page_icon="🧮")
st.title("🧠 Smart Calculator Pro")
st.caption("Every domain. One calculator. 💡")

# Initialize session state for input persistence
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

operation = st.selectbox("📌 Select Operation:", [
    "Addition", "Subtraction", "Multiplication", "Division",
    "Trigonometry", "Algebraic Equation", "Calculus Expression",
    "Statistics Summary", "Probability (event/total)", "Matrix Ops",
    "Unit Conversion", "Date Difference", "Simple Interest",
    "Graph Expression"
], key="operation_selector")

# --- Syntax Guidance ---
syntax_examples = {
    "Addition": "Example: `10 20 30`",
    "Subtraction": "Example: `100 25 10`",
    "Multiplication": "Example: `2 3 4`",
    "Division": "Example: `100 5 2`",
    "Trigonometry": "Example: `sin(pi/4)` (use `pi` for $\\pi$, `e` for Euler's number)",
    "Algebraic Equation": "Example: `x**2 + 5*x + 6 = 0` (ensure variable is `x`)",
    "Calculus Expression": "Example: `diff(x**2, x)` or `integrate(x**2, x)`",
    "Statistics Summary": "Example: `10 20 30 40` (space-separated numbers)",
    "Probability (event/total)": "Example: `100 25` (total_cases event_cases)",
    "Matrix Ops": "Example: `[[1,2],[3,4]]` (square matrices only, as a JSON-like list)",
    "Unit Conversion": "Example: `1000 g kg` or `25 celsius fahrenheit`",
    "Date Difference": "Example: `2025-01-01 2025-06-01` (YYYY-MM-DD)",
    "Simple Interest": "Example: `10000 5 2` (Principal Rate Time - ₹, %, Years)",
    "Graph Expression": "Example: `x**2 - 4*x + 3` (use `x` as the variable)"
}

st.markdown(f"**🧾 Syntax:** {syntax_examples.get(operation, 'Enter valid input')}\n")

col1, col2 = st.columns([0.8, 0.2])

with col1:
    user_input = st.text_input("✍️ Enter input:", value=st.session_state.user_input, key="current_user_input")

with col2:
    st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space
    if st.button("Clear Input"):
        st.session_state.user_input = ""
        st.rerun() # Rerun to clear the text_input field immediately

if user_input:
    # Update session state with current input
    st.session_state.user_input = user_input

    result = ""
    with st.spinner("Calculating..."):
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
            result = "❌ Unknown operation selected. Please choose from the list."

    # Display results based on whether it's an error or success
    if isinstance(result, str):
        if "❌" in result:
            st.error(result)
        elif "Graph rendered successfully" in result:
            st.info(result) # Info for graph success, as the graph is the primary output
        else:
            st.success(result)
    # else: # This case is less likely now that all solvers return strings
    #     st.success("✅ Operation completed.")

st.markdown("""
---
#### 💹 Supported Formats & Tips:
- **Basic Arithmetic:** Separate numbers with spaces (e.g., `10 20 30`)
- **Trigonometry:** Use standard functions like `sin`, `cos`, `tan`. Use `pi` for $\\pi$ and `e` for Euler's number.
    - Example: `sin(pi/4)`
- **Algebra:** Equations must contain an `=` sign and involve `x`.
    - Example: `x**2 + 5*x + 6 = 0`
- **Calculus:** Use `diff(expression, variable)` for differentiation and `integrate(expression, variable)` for integration.
    - Example: `diff(x**2, x)` or `integrate(x**2, x)`
- **Matrix Operations:** Enter as a Python-style list of lists. Only square matrices are supported.
    - Example: `[[1,2],[3,4]]`
- **Unit Conversion:** Format as `value from_unit to_unit`.
    - Examples: `1000 g kg`, `2.5 m ft`, `25 celsius fahrenheit`
- **Date Difference:** Use `YYYY-MM-DD` format for both dates.
    - Example: `2025-01-01 2025-06-01`
- **Simple Interest:** Format as `principal rate time` (Rate in percentage, Time in years).
    - Example: `10000 5 2` (₹10,000 at 5% for 2 years)
- **Graph Expression:** Enter an expression involving `x` to plot.
    - Example: `x**2 - 4*x + 3`
---
Made with ❤️ using Streamlit, NumPy, SymPy, and Matplotlib.
""")