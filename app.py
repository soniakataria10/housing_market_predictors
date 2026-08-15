import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from train_model import HousingMarketModel
import joblib
import os

class HousingMarketPredictorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏠 Housing Market Predictor")
        self.root.geometry("1000x700")
        
        # Style configuration
        self.setup_styles()
        
        # Initialize model
        self.model = HousingMarketModel()
        self.load_model()
        
        # Create UI components
        self.create_widgets()
        
        # ranges for each feature
        self.field_ranges = {
                    'size_sqft': {
                        'min': 200,
                        'max': 10000,
                        'type': 'float',
                        'error_msg': 'Size must be between 200 and 10,000 sq ft'
                    },
                    'total_bedrooms': {
                        'min': 1,
                        'max': 10,
                        'type': 'int',
                        'error_msg': 'Bedrooms must be between 0 and 10'
                    },
                    'total_bathrooms': {
                        'min': 1,
                        'max': 6,
                        'type': 'float',
                        'error_msg': 'Bathrooms must be between 0 and 6'
                    },
                    'house_age': {
                        'min': 0,
                        'max': 50,
                        'type': 'int',
                        'error_msg': 'Age must be between 0 and 50 years'
                    }
                }

    def setup_styles(self):
        """Configure the UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.bg_color = '#f0f4f8'
        self.primary_color = '#2c3e50'
        self.accent_color = '#3498db'
        self.success_color = '#27ae60'
        
        self.root.configure(bg=self.bg_color)
        
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists('models/model.pkl') and os.path.exists('models/preprocessor.pkl'):
                self.model.load_model('models/model.pkl', 'models/preprocessor.pkl')
                self.model_loaded = True
            else:
                self.model_loaded = False
                messagebox.showwarning(
                    "Model Not Found",
                    "Model files not found. Please train the model first using train_model.py"
                )
        except Exception as e:
            self.model_loaded = False
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
    
    def create_widgets(self):
        """Create all UI widgets"""
        # Main container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_container,
            text="🏠 House Price Predictor",
            font=('Arial', 24, 'bold'),
            foreground=self.primary_color
        )
        title_label.pack(pady=(0, 20))
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_container,
            text="Enter house details to get an estimated price prediction",
            font=('Arial', 12)
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_prediction_tab()
        self.create_data_tab()
        self.create_visualization_tab()
        
    def create_prediction_tab(self):
        """Create the prediction input tab"""
        prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(prediction_tab, text="🔮 Predict")
        
        # Input frame
        input_frame = ttk.LabelFrame(prediction_tab, text="House Details", padding="20")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create input fields
        fields = [
            ("Size (sq ft):", "size_sqft", "1500"),
            ("Total Bedrooms:", "total_bedrooms", "3"),
            ("Total Bathrooms:", "total_bathrooms", "2"),
            ("House Age (years):", "house_age", "5"),
            ("Location:", "location", ["Urban", "Suburban", "Rural"]),
            ("Garage:", "garage", ["Yes", "No"]),
            ("Condition:", "condition", ["Excellent", "Good", "Average", "Poor"])
        ]
        
        self.input_vars = {}
        
        # Create grid layout
        for i, (label_text, field_name, default) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            
            # Label
            label = ttk.Label(input_frame, text=label_text, font=('Arial', 11))
            label.grid(row=row, column=col, sticky='w', pady=8, padx=(0, 10))
            
            # Input field
            if isinstance(default, list):
                # Dropdown
                var = tk.StringVar(value=default[0])
                dropdown = ttk.Combobox(
                    input_frame,
                    textvariable=var,
                    values=default,
                    state='readonly',
                    width=20
                )
                dropdown.grid(row=row, column=col+1, sticky='w', pady=8)
            else:
                # Text entry
                var = tk.StringVar(value=default)
                entry = ttk.Entry(input_frame, textvariable=var, width=23)
                entry.grid(row=row, column=col+1, sticky='w', pady=8)
            
            self.input_vars[field_name] = var
        
        # Prediction button and result
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=30)
        
        predict_btn = ttk.Button(
            button_frame,
            text="💰 Predict Price",
            command=self.predict_price,
            style='Accent.TButton'
        )
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_inputs
        )
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Result display
        self.result_frame = ttk.LabelFrame(input_frame, text="Prediction Result", padding="20")
        self.result_frame.grid(row=4, column=0, columnspan=4, sticky='ew', pady=20)
        
        self.result_label = ttk.Label(
            self.result_frame,
            text="Enter house details and click 'Predict Price'",
            font=('Arial', 14),
            foreground='gray'
        )
        self.result_label.pack(pady=10)
        
        # Configure grid weights
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)
        
    def create_data_tab(self):
        """Create the data viewer tab"""
        data_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="📊 Data")
        
        # Data display
        data_frame = ttk.LabelFrame(data_tab, text="Sample Data", padding="10")
        data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create Treeview for data display
        columns = ('Size', 'Total Bedrooms', 'Total Bathrooms', 'House Age', 'Location', 'Garage', 'Condition', 'Price')
        self.tree = ttk.Treeview(data_frame, columns=columns, show='headings', height=10)
        
        # Set column headings
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Add scrollbars
        scrollbar_y = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(data_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Pack treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar_y.grid(row=0, column=1, sticky='ns')
        scrollbar_x.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        data_frame.grid_rowconfigure(0, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)
        
        # Load data
        self.load_data_preview()
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(data_tab, text="Data Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=8, width=80)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.load_statistics()
        
    def create_visualization_tab(self):
        """Create the visualization tab"""
        viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(viz_tab, text="📈 Visualizations")
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load visualizations
        self.load_visualizations()
        
    def predict_price(self):
        """Handle price prediction"""
        if not self.model_loaded:
            messagebox.showerror(
                "Model Not Loaded",
                "Please train the model first using train_model.py"
            )
            return
        
        try:
            # Collect input values
            features = {}
            
            for field, var in self.input_vars.items():
                value = var.get().strip()
                
                if field in self.field_ranges:
                    if value == '':
                        messagebox.showerror("Input Error", f"Please enter {field}")
                        return
                    try:
                        num_value = float(value)
                    except ValueError:
                        messagebox.showerror("Input Error", f"Invalid numeric value for {field}")
                        return
                    
                    if self.field_ranges[field]['type'] == 'int':
                        if num_value != int(num_value):
                            messagebox.showerror("Input Error", f"{field} must be a whole number.")
                            return
                    num_value = int(num_value)
                    features[field] = num_value

                    field_limit = self.field_ranges[field]
                    if num_value > field_limit['max'] or num_value < field_limit['min']:
                        messagebox.showerror("Input Error", f"Please enter {field} between {field_limit['min']} and {field_limit['max']}.")
                        return
                    
                else:
                    features[field] = value

            # Make prediction
            price = self.model.predict(features)
            
            # Display result with animation
            self.display_result(price, features)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
    
    def display_result(self, price, features):
        """Display prediction result with details"""
        # Clear previous result
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Format price
        price_formatted = f"${price:,.2f}"
        
        # Main result
        result_label = ttk.Label(
            self.result_frame,
            text=f"Predicted House Price: {price_formatted}",
            font=('Arial', 18, 'bold'),
            foreground=self.success_color
        )
        result_label.pack(pady=0)  #it was pady 10 before
        
        # Additional details
        details_text = f"""
        📋 House Details:
        • Size: {features.get('size_sqft', 'N/A')} sq ft
        • Total Bedrooms: {features.get('total_bedrooms', 'N/A')}
        • Total Bathrooms: {features.get('total_bathrooms', 'N/A')}
        • House Age: {features.get('house_age', 'N/A')} years
        • Location: {features.get('location', 'N/A')}
        • Garage: {features.get('garage', 'N/A')}
        • Condition: {features.get('condition', 'N/A')}
        """
        
        details_label = ttk.Label(
            self.result_frame,
            text=details_text,
            font=('Arial', 11),
            justify='left'
        )
        details_label.pack(pady=0) #it was pady 10 before
        
        # Confidence indicator
        confidence_frame = ttk.Frame(self.result_frame)
        confidence_frame.pack(pady=10)
        
        ttk.Label(confidence_frame, text="Confidence: High", font=('Arial', 10)).pack()
    
    def clear_inputs(self):
        """Clear all input fields"""
        for field, var in self.input_vars.items():
            if field in ['size_sqft', 'total_bedrooms', 'total_bathrooms', 'house_age']:
                var.set('')
            else:
                # Reset dropdowns to first value
                dropdown_values = {
                    'location': ['Urban', 'Suburban', 'Rural'],
                    'garage': ['Yes', 'No'],
                    'condition': ['Excellent', 'Good', 'Average', 'Poor']
                }
                if field in dropdown_values:
                    var.set(dropdown_values[field][0])
        
        # Clear result
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        result_label = ttk.Label(
            self.result_frame,
            text="Enter house details and click 'Predict Price'",
            font=('Arial', 14),
            foreground='gray'
        )
        result_label.pack(pady=10)
    
    def load_data_preview(self):
        """Load preview of data into treeview"""
        try:
            if os.path.exists('data/ontario_housing.csv'):
                df = pd.read_csv('data/ontario_housing.csv')
                
                # Clear existing items
                for item in self.tree.get_children():
                    self.tree.delete(item)
                
                # Add first 10 rows
                for _, row in df.head(104).iterrows():
                    values = [
                        f"{row['size_sqft']:.0f}",
                        str(row['total_bedrooms']),
                        str(row['total_bathrooms']),
                        str(row['house_age']),
                        row['location'],
                        row['garage'],
                        row['condition'],
                        f"${row['price']:,.0f}"
                    ]
                    self.tree.insert('', 'end', values=values)
        except Exception as e:
            print(f"Error loading data preview: {e}")
    
    def load_statistics(self):
        """Load data statistics"""
        try:
            if os.path.exists('data/ontario_housing.csv'):
                df = pd.read_csv('data/ontario_housing.csv')
                
                stats = df.describe()
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, stats.to_string())
                self.stats_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    def load_visualizations(self):
        """Load visualizations"""
        try:
            if os.path.exists('feature_importance.png'):
                # Clear axes
                self.ax1.clear()
                self.ax2.clear()
                
                # Load and display image
                img = plt.imread('feature_importance.png')
                self.ax1.imshow(img)
                self.ax1.axis('off')
                self.ax1.set_title('Feature Importance')
                
                # Create scatter plot of price vs size
                if os.path.exists('data/ontario_housing.csv'):
                    df = pd.read_csv('data/ontario_housing.csv')
                    self.ax2.scatter(df['size_sqft'], df['price'], alpha=0.5)
                    self.ax2.set_xlabel('Size (sq ft)')
                    self.ax2.set_ylabel('Price ($)')
                    self.ax2.set_title('Price vs Size')
                
                self.canvas.draw()
        except Exception as e:
            print(f"Error loading visualizations: {e}")

def main():
    root = tk.Tk()
    app = HousingMarketPredictorUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()