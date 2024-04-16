import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import IncrementalPCA
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import os


# Function to establish database connection
def establish_connection():
    try:
        # Establish connection
        conn = psycopg2.connect(
            dbname="ECG_EDA",
            user="postgres",
            password="3234",  # Replace with your password
            host="localhost",
            port="5432"
        )
        return conn
    except psycopg2.Error as e:
        print("Error connecting to PostgreSQL:", e)
        return None

        
# Function to fetch data from PostgreSQL
def fetch_data(query, conn):
    cur = conn.cursor()
    try:
        # Execute query
        cur.execute(query)
        data = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(data, columns=columns)
    finally:
        # Close cursor
        cur.close()



# Function to visualize PCA using Incremental PCA
def visualize_pca_incremental():
    # Get the current directory
    current_dir = os.getcwd()

    # Construct the path to the plots folder
    plots_folder_path = os.path.join(current_dir, "plots")

    # Check if the plots folder exists
    if not os.path.exists(plots_folder_path):
        st.error("Plots folder not found.")
        return

    # Construct the paths to the images
    image1_path = os.path.join(plots_folder_path, "with_PCA.png")

    # Check if the image exists
    if not os.path.exists(image1_path):
        st.error("Image not found.")
        return

    # Display the image
    st.image([image1_path], caption=['with_PCA.png'])

def visualize_without_pca_incremental():
    # Get the current directory
    current_dir = os.getcwd()

    # Construct the path to the plots folder
    plots_folder_path = os.path.join(current_dir, "plots")

    # Check if the plots folder exists
    if not os.path.exists(plots_folder_path):
        st.error("Plots folder not found.")
        return

    # Construct the paths to the images
    image1_path = os.path.join(plots_folder_path, "without_PCA.png")

    # Check if the image exists
    if not os.path.exists(image1_path):
        st.error("Image not found.")
        return

    # Display the image
    st.image([image1_path], caption=['without_PCA.png'])



# Function to plot total patients with and without arrhythmia
def plot_total_patients_with_without_arrhythmia(conn):
    cur = conn.cursor()
    try:
        # Fetch patient counts from the database
        cur.execute("""
            SELECT dxname, COUNT(*) FROM ECG GROUP BY dxname;
        """)
        rows = cur.fetchall()

        # Calculate counts of patients with and without arrhythmia
        arrhythmia_count = 0
        no_arrhythmia_count = 0
        for row in rows:
            if row[0] == 'NOF':
                no_arrhythmia_count += row[1]
            else:
                arrhythmia_count += row[1]

        # Plotting
        labels = ['No Arrhythmia', 'Arrhythmia']
        counts = [no_arrhythmia_count, arrhythmia_count]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, counts, color=['skyblue', 'salmon'])
        plt.title('Number of Patients with and without Arrhythmia')
        plt.xlabel('Arrhythmia Status')
        plt.ylabel('Number of Patients')

        # Adding annotations
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, '%d' % int(height), ha='center', va='bottom')

        st.pyplot(plt.gcf())

    finally:
        # Close cursor
        cur.close()

# Function to plot heart rate distribution
def plot_heart_rate_distribution(conn):
    cur = conn.cursor()
    try:
        if conn is not None:
            cur.execute("SELECT hr, dxname FROM ecg")
            rows = cur.fetchall()

            # Fetch data into a DataFrame
            df = pd.DataFrame(rows, columns=['hr', 'dxname'])

            # Replace 'NOF' with 'No Disease' and others with 'Disease'
            df['dxname'] = df['dxname'].apply(lambda x: 'No Disease' if x == 'NOF' else 'Disease')

            # Create subplots
            fig, axes = plt.subplots(2, 1, figsize=(10, 12))

            # Plot histogram with arrhythmias
            sns.histplot(df[df['dxname'] == 'Disease']['hr'], ax=axes[0], color='skyblue', kde=True)
            axes[0].set_title('Heart Rate Distribution with Arrhythmias')
            axes[0].set_xlabel('Heart Rate')
            axes[0].set_ylabel('Frequency')
            
            # Add mean and median lines to the plot
            mean_arrhythmias = df[df['dxname'] == 'Disease']['hr'].mean()
            median_arrhythmias = df[df['dxname'] == 'Disease']['hr'].median()
            axes[0].axvline(mean_arrhythmias, color='blue', linestyle='dashed', linewidth=1, label=f'Mean: {mean_arrhythmias:.2f}')
            axes[0].axvline(median_arrhythmias, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_arrhythmias:.2f}')
            axes[0].legend()

            # Plot histogram without arrhythmias
            sns.histplot(df[df['dxname'] == 'No Disease']['hr'], ax=axes[1], color='salmon', kde=True)
            axes[1].set_title('Heart Rate Distribution without Arrhythmias')
            axes[1].set_xlabel('Heart Rate')
            axes[1].set_ylabel('Frequency')

            # Add mean and median lines to the plot
            mean_no_arrhythmias = df[df['dxname'] == 'No Disease']['hr'].mean()
            median_no_arrhythmias = df[df['dxname'] == 'No Disease']['hr'].median()
            axes[1].axvline(mean_no_arrhythmias, color='blue', linestyle='dashed', linewidth=1, label=f'Mean: {mean_no_arrhythmias:.2f}')
            axes[1].axvline(median_no_arrhythmias, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_no_arrhythmias:.2f}')
            axes[1].legend()

            plt.tight_layout()
            st.pyplot(plt.gcf())

    finally:
        # Close cursor
        cur.close()
        
def plot_gender_distribution(conn):
    if conn is not None:
        cur = conn.cursor()
        try:
            cur.execute("SELECT gender, dxname FROM ecg")
            rows = cur.fetchall()

            # Fetch data into a DataFrame
            df = pd.DataFrame(rows, columns=['gender', 'dxname'])

            # Replace 'NOF' with 'No Disease' and others with 'Disease'
            df['dxname'] = df['dxname'].apply(lambda x: 'No Disease' if x == 'NOF' else 'Disease')

            # Plot distribution
            plt.figure(figsize=(8, 6))
            sns.countplot(x='gender', hue='dxname', data=df)

            # Add annotations
            for p in plt.gca().patches:
                height = p.get_height()
                plt.gca().text(p.get_x() + p.get_width() / 2, height + 0.1, f"{height:,}", ha='center')

            plt.title('Distribution of Gender with and without Arrhythmias')
            plt.xlabel('Gender')
            plt.ylabel('Count')
            plt.legend(title='Arrhythmias')
            st.pyplot(plt.gcf())

        finally:
            # Close cursor
            cur.close()


# Function to plot age distribution
def plot_age_distribution(conn):
    if conn is not None:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT age, dxname FROM ecg")
            rows = cursor.fetchall()

            # Fetch data into a DataFrame
            df = pd.DataFrame(rows, columns=['age', 'dxname'])

            # Replace 'NOF' with 'No Disease' and others with 'Disease'
            df['dxname'] = df['dxname'].apply(lambda x: 'No Disease' if x == 'NOF' else 'Disease')

            # Remove rows where age is 0
            df = df[df['age'] != 0]

            # Create separate plots for data with and without arrhythmia
            fig_with_arrhythmia = px.histogram(df[df['dxname'] == 'Disease'], x='age', nbins=20,
                                                title='Age Distribution with Arrhythmias', labels={'age': 'Age', 'count': 'Count'})
            fig_without_arrhythmia = px.histogram(df[df['dxname'] == 'No Disease'], x='age', nbins=20,
                                                   title='Age Distribution without Arrhythmias', labels={'age': 'Age', 'count': 'Count'})

            st.plotly_chart(fig_with_arrhythmia)
            st.plotly_chart(fig_without_arrhythmia)

        finally:
            # Close cursor
            cursor.close()
            

# Function to plot correlation heatmap
def plot_correlation_heatmap(conn):
    if conn is not None:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT age, gender, hr, dxname FROM ecg")
            rows = cursor.fetchall()

            # Fetch data into a DataFrame
            df = pd.DataFrame(rows, columns=['age', 'gender', 'hr', 'dxname'])

            # Replace 'NOF' with 0 (No Disease) and others with 1 (Disease)
            df['dxname'] = df['dxname'].apply(lambda x: 0 if x == 'NOF' else 1)

            # Convert categorical variables to numerical for correlation analysis
            df['gender'] = df['gender'].map({'M': 0, 'F': 1})

            # Calculate correlation matrix
            corr_matrix = df.corr()

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            heatmap = sns.heatmap(corr_matrix, annot=True, cmap='cividis', fmt=".2f", linewidths=0.5)

            # Customize axis labels
            heatmap.set_xticklabels(['Age', 'Gender', 'Heart Rate', 'Diseases'])
            heatmap.set_yticklabels(['Age', 'Gender', 'Heart Rate', 'Diseases'])

            plt.title('Correlation Matrix of Features')
            st.pyplot(plt.gcf())

        finally:
            # Close cursor
            cursor.close()
            

# Function to plot top 5 arrhythmia diseases
def plot_top_arrhythmia_diseases(conn):
    if conn is not None:
        try:
            # SQL query to retrieve the top 5 arrhythmia diseases (excluding 'NOF') and their counts
            query = """
                SELECT dxname, COUNT(*) as disease_count 
                FROM ecg 
                WHERE dxname != 'NOF' 
                GROUP BY dxname 
                ORDER BY disease_count DESC 
                LIMIT 5
            """

            # Execute the query and fetch results into a DataFrame
            df = pd.read_sql_query(query, conn)

            # Filter out 'NOF' category
            df_filtered = df[df['dxname'] != 'NOF']

            # Create an interactive pie plot using Plotly
            fig = px.pie(df_filtered, values='disease_count', names='dxname', title='Top 5 Arrhythmia Diseases')
            st.plotly_chart(fig)

        except psycopg2.Error as e:
            st.error("Error executing SQL query:", e)

# Function to plot top 5 arrhythmia diseases by gender
def plot_top_arrhythmia_diseases_by_gender(conn):
    if conn is not None:
        try:
            # SQL query to retrieve the top 5 arrhythmia diseases for male and female separately (excluding 'NOF') and their counts
            query_male = """
                SELECT dxname, COUNT(*) as disease_count 
                FROM ecg 
                WHERE dxname != 'NOF' AND gender = 'Male' 
                GROUP BY dxname 
                ORDER BY disease_count DESC 
                LIMIT 5
            """

            query_female = """
                SELECT dxname, COUNT(*) as disease_count 
                FROM ecg 
                WHERE dxname != 'NOF' AND gender = 'Female' 
                GROUP BY dxname 
                ORDER BY disease_count DESC 
                LIMIT 5
            """

            # Execute the queries and fetch results into DataFrames
            df_male = pd.read_sql_query(query_male, conn)
            df_female = pd.read_sql_query(query_female, conn)

            # Create subplots for male and female arrhythmia diseases
            fig = go.Figure()

            # Add subplot for male
            fig.add_trace(go.Bar(
                x=df_male['dxname'],
                y=df_male['disease_count'],
                name='Male',
                marker_color='blue'
            ))

            # Add subplot for female
            fig.add_trace(go.Bar(
                x=df_female['dxname'],
                y=df_female['disease_count'],
                name='Female',
                marker_color='pink'
            ))

            # Update layout
            fig.update_layout(
                title='Top 5 Arrhythmia Diseases by Gender',
                xaxis_title='Arrhythmia Disease',
                yaxis_title='Disease Count',
                barmode='group'
            )

            # Show plot
            st.plotly_chart(fig)

        except psycopg2.Error as e:
            st.error("Error executing SQL query:", e)
             

# Function to plot top 5 arrhythmia diseases by age group
def plot_top_arrhythmia_diseases_by_age_group(conn):
    if conn is not None:
        try:
            # Define query to fetch data
            query = """
            SELECT age, dxname
            FROM ecg
            WHERE dxname != 'NOF' AND age >= 4;
            """

            # Execute query and fetch results
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            # Close communication with the database
            cursor.close()

            # Convert data to pandas dataframe
            df = pd.DataFrame(rows, columns=["age", "dxname"])

            # Define age groups
            age_groups = [(0, 20), (20, 40), (40, 60), (60, 80), (80, float('inf'))]

            # Create grouped bar charts for each age group
            for age_group in age_groups:
                age_start, age_end = age_group
                df_age_group = df[(df['age'] >= age_start) & (df['age'] < age_end)]
                cross_tab = pd.crosstab(df_age_group['age'], df_age_group['dxname'])
                top_diseases = cross_tab.sum().nlargest(5).index
                cross_tab_top = cross_tab[top_diseases]

                # Create traces for each disease
                traces = []
                for disease in cross_tab_top.columns:
                    trace = go.Bar(
                        x=cross_tab_top.index,
                        y=cross_tab_top[disease],
                        name=disease
                    )
                    traces.append(trace)

                # Create layout for the plot
                layout = go.Layout(
                    title=f'Top 5 Diseases for Age Group {age_start}-{age_end}',
                    xaxis=dict(title='Age'),
                    yaxis=dict(title='Count'),
                    barmode='group'
                )

                # Create figure and plot
                fig = go.Figure(data=traces, layout=layout)
                st.plotly_chart(fig)

        except psycopg2.Error as e:
            st.error("Error executing SQL query:", e)
            
 
 
# Function to plot top 5 arrhythmia diseases by age group
def plot_top_arrhythmia_diseases_by_age_group(conn):
    if conn is not None:
        try:
            # Define query to fetch data
            query = """
            SELECT age, dxname
            FROM ecg
            WHERE dxname != 'NOF' AND age >= 4;
            """

            # Execute query and fetch results
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            # Close communication with the database
            cursor.close()

            # Convert data to pandas dataframe
            df = pd.DataFrame(rows, columns=["age", "dxname"])

            # Define age groups
            age_groups = [(0, 20), (20, 40), (40, 60), (60, 80), (80, float('inf'))]

            # Create grouped bar charts for each age group
            for age_group in age_groups:
                age_start, age_end = age_group
                df_age_group = df[(df['age'] >= age_start) & (df['age'] < age_end)]
                cross_tab = pd.crosstab(df_age_group['age'], df_age_group['dxname'])
                top_diseases = cross_tab.sum().nlargest(5).index
                cross_tab_top = cross_tab[top_diseases]

                # Create traces for each disease
                traces = []
                for disease in cross_tab_top.columns:
                    trace = go.Bar(
                        x=cross_tab_top.index,
                        y=cross_tab_top[disease],
                        name=disease
                    )
                    traces.append(trace)

                # Create layout for the plot
                layout = go.Layout(
                    title=f'Top 5 Diseases for Age Group {age_start}-{age_end}',
                    xaxis=dict(title='Age'),
                    yaxis=dict(title='Count'),
                    barmode='group'
                )

                # Create figure and plot
                fig = go.Figure(data=traces, layout=layout)
                st.plotly_chart(fig)

        except psycopg2.Error as e:
            st.error("Error executing SQL query:", e)

 # Function to plot t-SNE visualization
def plot_tsne_visualization(conn):
    if conn is not None:
        try:
            # Fetch data from the database
            cursor = conn.cursor()
            cursor.execute("SELECT age, gender, hr, dxname FROM ecg")
            rows = cursor.fetchall()

            # Fetch data into a DataFrame
            df = pd.DataFrame(rows, columns=['age', 'gender', 'hr', 'dxname'])

            # Replace 'NOF' with 0 (No Disease) and others with 1 (Disease)
            df['dxname'] = df['dxname'].apply(lambda x: 0 if x == 'NOF' else 1)

            # One-hot encode gender column
            df = pd.get_dummies(df, columns=['gender'], drop_first=True)

            # Separate data for patients with and without arrhythmias
            arrhythmias_df = df[df['dxname'] == 1]
            no_arrhythmias_df = df[df['dxname'] == 0]

            # Drop rows with missing values
            arrhythmias_df = arrhythmias_df.dropna()
            no_arrhythmias_df = no_arrhythmias_df.dropna()

            # Perform t-SNE dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42)
            arrhythmias_tsne = tsne.fit_transform(arrhythmias_df.drop(columns=['dxname']))
            no_arrhythmias_tsne = tsne.fit_transform(no_arrhythmias_df.drop(columns=['dxname']))

            # Plot t-SNE visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(arrhythmias_tsne[:, 0], arrhythmias_tsne[:, 1], color='skyblue', label='With Arrhythmias')
            plt.scatter(no_arrhythmias_tsne[:, 0], no_arrhythmias_tsne[:, 1], color='salmon', label='Without Arrhythmias')
            plt.title('t-SNE Visualization of Patients with and without Arrhythmias')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.legend()
            st.pyplot(plt.gcf())

        except psycopg2.Error as e:
            st.error("Error executing SQL query:", e)        
            
# Main function
def main():
    st.title("ECG Analysis")

    # Establish database connection
    conn = establish_connection()

   
    # Dropdown for EDA plots
    # Dropdown for EDA plots
    selected_plot = st.selectbox(
        "Select EDA Plot",
        ("Total Patients with and without Arrhythmia", "Heart Rate Distribution", "Age Distribution", "Top 5 Arrhythmia Diseases", "Top 5 Arrhythmia Diseases by Gender", "Top 5 Arrhythmia Diseases by Age Group", "ECG 3D plot Before PCA", "ECG 3D Plot after PCA", "Data Points after TSNE")
    )

    if selected_plot == "ECG 3D Plot after PCA":   
        # visualize_pca_incremental(data_generator, conn, batch_size=2000)
        visualize_pca_incremental()

    elif selected_plot == "ECG 3D plot Before PCA":
        visualize_without_pca_incremental()

    # Display corresponding EDA plot based on selection
    elif selected_plot == "Total Patients with and without Arrhythmia":
        plot_total_patients_with_without_arrhythmia(conn)
        
    elif selected_plot == "Heart Rate Distribution":
        plot_heart_rate_distribution(conn)
        
    elif selected_plot == "Age Distribution":
        plot_age_distribution(conn)
            
    elif selected_plot == "Data Points after TSNE":
        plot_tsne_visualization(conn)
    
    elif selected_plot == "Top 5 Arrhythmia Diseases":
        plot_top_arrhythmia_diseases(conn)
    
    elif selected_plot == "Top 5 Arrhythmia Diseases by Gender":
        plot_top_arrhythmia_diseases_by_gender(conn)
    
    elif selected_plot == "Top 5 Arrhythmia Diseases by Age Group":
        plot_top_arrhythmia_diseases_by_age_group(conn)

    # Close connection
    if conn is not None:
        conn.close()

if __name__ == "__main__":
    main()