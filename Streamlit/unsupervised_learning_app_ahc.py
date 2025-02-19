import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df = pd.read_csv("Data/ClusteredData.csv", index_col=0)
orig_cols = ['Age', 'Salary', 'Total Spent', 'Payment Method', 'Days Since Joined',
        'Gender', 'First Name', 'Last Name', 'Email', 'IBAN']

for k in range(2,10):
    df[f"{k}-Clusters Group"] = df[f"{k}-Clusters Group"].astype(str)

def scatter(df, k):
      
    if k == 1:
        scatter_fig = px.scatter_matrix(
            df, 
            dimensions=['Age','Salary','Total Spent','Days Since Joined'],
            hover_data=orig_cols,
            height=700,
            color_discrete_sequence=['#41B6C4']  # Force same palette even if no color variable is provided
        )
        scatter_fig.update_layout(
            title_text="Scatter Matrix of Numerical Relationships"
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        # Ensure cluster labels are treated as strings for consistent color mapping
        df[f"{k}-Clusters Group"] = df[f"{k}-Clusters Group"].astype(str)
        df = df.sort_values(by=f"{k}-Clusters Group")
        scatter_fig = px.scatter_matrix(
            df, 
            dimensions=['Age','Salary','Total Spent','Days Since Joined'],
            hover_data=orig_cols,
            height=700,
            color=f"{k}-Clusters Group",
            color_discrete_sequence=px.colors.qualitative.Plotly  # Use same color sequence as barplots/histograms
        )
        scatter_fig.update_layout(
            title_text="Scatter Matrix of Numerical Relationships by Cluster"
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

def hists(df, k):

    numerical_columns = ['Age', 'Salary', 'Total Spent', 'Days Since Joined']
    
    # Compute bin settings for each numerical column
    desired_bins = 30
    bin_settings = {}
    for col in numerical_columns:
        x_min = df[col].min()
        x_max = df[col].max()
        bin_width = (x_max - x_min) / desired_bins
        bin_settings[col] = {"start": x_min, "end": x_max, "size": bin_width}
    
    if k == 1:
        cols = len(numerical_columns)
        rows = k
        fig = make_subplots(rows=k, cols=cols, subplot_titles=numerical_columns)
        for i, col in enumerate(numerical_columns):
            # Determine the current row and column in the subplot grid
            row = (i // cols) + 1
            col_position = (i % cols) + 1

            # Add histogram to the subplot with fixed bin settings for that variable
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    name=col,
                    marker_color='#41B6C4',
                    xbins=bin_settings[col]
                ),
                row=row, col=col_position
            )

        # Update layout for better appearance
        fig.update_layout(
            title_text="Histograms of Numerical Columns",
            height=300 * rows,  # Adjust height based on the number of rows
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        clusters = sorted(df[f'{k}-Clusters Group'].unique())
        fig = make_subplots(
            rows=k, 
            cols=len(numerical_columns), 
            subplot_titles=[f"Cluster {cluster} - {col}" for cluster in clusters for col in numerical_columns]
        )
        # Define a discrete color sequence (one color per cluster)
        colors = px.colors.qualitative.Plotly

        # Loop through each cluster (row) and each numerical column (column)
        for row_idx, cluster in enumerate(clusters, start=1):
            # Filter the dataframe for this cluster
            df_cluster = df[df[f'{k}-Clusters Group'] == cluster]
            # Pick a color for this cluster (cycling if needed)
            color = colors[(row_idx - 1) % len(colors)]
            
            for col_idx, col in enumerate(numerical_columns, start=1):
                fig.add_trace(
                    go.Histogram(
                        x=df_cluster[col],
                        name=f"Cluster {cluster}",
                        marker_color=color,
                        xbins=bin_settings[col],  # Use same bin settings for this variable
                        showlegend=False
                    ),
                    row=row_idx,
                    col=col_idx
                )
                
        # Optionally, update x-axis ranges to match the global range (this is already inherent with xbins)
        for col_idx, col in enumerate(numerical_columns, start=1):
            x_min = bin_settings[col]['start']
            x_max = bin_settings[col]['end']
            for row_idx in range(1, k + 1):
                fig.update_xaxes(range=[x_min, x_max], row=row_idx, col=col_idx)
                
        # Update layout for better appearance
        fig.update_layout(
            title_text="Histograms of Numerical Columns by Cluster",
            height=300 * k,  # Adjust height based on number of clusters
            barmode='overlay'  # Use 'overlay' so histograms don't shift
        )
        st.plotly_chart(fig, use_container_width=True)



def bars(df, k):


    categorical_columns = ['Payment Method', 'Gender']
    cols = len(categorical_columns)
    
    if k == 1:
        rows = 1
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=categorical_columns)
        for i, col in enumerate(categorical_columns):
            # For k==1, all plots go into row 1
            row = 1
            col_position = i + 1

            # Calculate value counts for the categorical column
            value_counts = df[col].value_counts()

            # Add horizontal bar plot to the subplot
            fig.add_trace(
                go.Bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    name=col,
                    orientation='h',
                    marker_color='#41B6C4'
                ),
                row=row,
                col=col_position
            )

        # Set x-axis range based on global maximum value count per variable
        for col_idx, col in enumerate(categorical_columns, start=1):
            max_count = df[col].value_counts().max()
            fig.update_xaxes(range=[0, max_count * 1.1], row=1, col=col_idx)

        fig.update_layout(
            title_text="Barplots of Categorical Variables",
            height=300,  # one row
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # For k > 1, we assume a column f'{k}-Clusters Group' exists.
        clusters = sorted(df[f'{k}-Clusters Group'].unique())
        rows = k
        subplot_titles = [f"Cluster {cluster} - {col}" for cluster in clusters for col in categorical_columns]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        colors = px.colors.qualitative.Plotly

        # Loop through each cluster (row) and each categorical variable (column)
        for row_idx, cluster in enumerate(clusters, start=1):
            df_cluster = df[df[f'{k}-Clusters Group'] == cluster]
            color = colors[(row_idx - 1) % len(colors)]
            for col_idx, col in enumerate(categorical_columns, start=1):
                value_counts = df_cluster[col].value_counts()
                fig.add_trace(
                    go.Bar(
                        x=value_counts.values,
                        y=value_counts.index,
                        orientation='h',
                        marker_color=color,
                        name=f"Cluster {cluster}",
                        showlegend=False
                    ),
                    row=row_idx,
                    col=col_idx
                )

        # For each categorical variable, compute the maximum count across clusters
        for col_idx, col in enumerate(categorical_columns, start=1):
            max_count = 0
            for cluster in clusters:
                cluster_max = df[df[f'{k}-Clusters Group'] == cluster][col].value_counts().max()
                if cluster_max > max_count:
                    max_count = cluster_max
            # Update the x-axis for every subplot in that column
            for row_idx in range(1, rows + 1):
                fig.update_xaxes(range=[0, max_count * 1.1], row=row_idx, col=col_idx)

        fig.update_layout(
            title_text="Barplots of Categorical Variables by Cluster",
            height=300 * rows,
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)



def plot_data(df, vis, k = 1):
    if "Table" in vis:
        st.write("**Data Preview of 7 Random Customers**")
        st.dataframe(df[orig_cols].sample(7))
    if "Scatter Matrix" in vis:
        scatter(df,k)
    if "Histograms" in vis:
        hists(df,k)
    if "Barplots" in vis:
        bars(df,k)

def main():
    HORIZONTAL = "Logo.png"
    ICON = "Icon.png"

    
    st.logo(HORIZONTAL, size="large", link="http://www.krisolis.ie", icon_image=ICON)
    st.set_page_config(
        page_title="Krisolis Unsupervised Learning Demonstration",
        page_icon=ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': """# Krisolis Unsupervised Learning Demonstration. 
            Created by Eoghan Staunton @Krisolis
            Using scikit-learn and plotly"""
        })

    st.title("Unsupervised Learning")
    st.subheader("Demonstration with Clustering")
    st.markdown(
    """
    - Use the controls in the sidebar to choose the number of clusters to segment the customers into.
    - Visualise the results and discuss what you think is the optimal number of clusters. You can choose the visualisations displayed in the sidebar.
    - What does each cluster represent?"""
    )
    
    if "k" not in st.session_state:
        st.session_state["k"] = 1

    st.session_state["k"] = st.sidebar.slider("Number of Clusters",1,9,1,1)
    st.session_state["vis"] = st.sidebar.multiselect("Visualisations to Display",("Table", "Scatter Matrix", "Barplots", "Histograms"), default=["Table","Scatter Matrix"])

    st.header("Data Visualisation",help="Choose which visualisations are displayed in the sidebar. Hover over the visualisations to view more details.")
    
    plot_data(df, st.session_state["vis"], st.session_state["k"])
    

if __name__ == "__main__":
    main()