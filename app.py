# Basic code to run streamlit app
import streamlit as st

def main():
    st.title("Recommendation System")
    st.subheader("The recommendation systems presented here is based on the collaborative filtering algorithm. It uses the overview of the item to recommend the similar items.")
    st.write("This is the page to test students on the recommendation system and how to deploy a user-friendly interface for it. Click each of the pages to see the different propositions.")
    st.sidebar.title("RS-CF-APP")

if __name__ == '__main__':
    main()