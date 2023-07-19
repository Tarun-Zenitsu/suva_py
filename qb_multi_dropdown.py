import openai
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import datetime

# Create a temporary database to store the inputs
temp_db = create_engine('sqlite:///temp.db', echo=True)


class WordCounter:
    def __init__(self):
        self.total_words = 0
        self.word_limit = 1000

    def count_words(self, text):
        words = text.split()
        word_count = len(words)
        if self.total_words + word_count > self.word_limit:
            raise Exception("Word limit exceeded!")
        self.total_words += word_count

    def get_total_words(self):
        return self.total_words

    def get_remaining_words(self):
        return self.word_limit - self.total_words


# Function to create table definition prompt
def create_table_definition_prompt(df):
    prompt = '''### sqlite SQL table, with its properties:
#
# Sales({})
#
'''.format(",".join(str(x) for x in df.columns))
    return prompt


# Function to combine prompts
def combine_prompts(df, query_prompt):
    definition = create_table_definition_prompt(df)
    query_init_string = f"### A query to answer: {query_prompt}\nSELECT"
    return definition + query_init_string


# Function to execute SQL query
def get_sql(nlp_text, df, counter):
    print("********************************************************************")
    print(nlp_text)
    counter.count_words(nlp_text)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=combine_prompts(df, nlp_text),
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"]
    )
    query = response["choices"][0]["text"]
    if query.startswith(" "):
        query = "SELECT" + query

    # Add the operator and additional input to the query
    # if operator and column_name:
    # query += f" WHERE {column_name} {operator}"
    # if additional_date is not None:
    # query +=  f" {additional_date}"

    print(query)
    with temp_db.connect() as conn:
        result = conn.execute(text(query))
    return result.all()


# Read CSV file into DataFrame
def read_csv_file2(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")  # Specify the correct encoding here
        return df
    except Exception as e:
        st.error(f"Error occurred while reading the CSV file: {str(e)}")
        return None


def read_csv_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")
        with temp_db.connect() as conn:
            df.to_sql(name='Sales', con=conn, if_exists='replace', index=False)
        return df
    except Exception as e:
        st.error(f"Error occurred while reading the CSV file: {str(e)}")
        return None


# Function to clear the session state
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# Configure Streamlit page layout and title
st.set_page_config(
    page_title="Query Builder",
    layout="wide",
    page_icon=":bar_chart:"
)

# Set OpenAI API key
openai.api_key = "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"

# Add CSS styles
st.markdown(
    """
    <style>
    .front-page {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .saved-input {
        display: flex;
        align-items: center;
    }
    .saved-input .remove-input {
        margin-left: 10px;
        color: red;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Front page content
st.title(":bar_chart: Query Builder")
st.markdown('<div class="front-page">', unsafe_allow_html=True)
st.title("Please write your *Query* Here :arrow_down_small:")

# Sidebar - CSV file upload and refresh button
uploaded_file = st.sidebar.file_uploader("Upload CSV File")
refresh_button = st.sidebar.button("Refresh")

# Clear session state on refresh button click
if refresh_button:
    clear_session_state()

# Initialize variables for first input and button click
first_input = ""
button_clicked = False

# Retrieve or create the session state
if 'input_list' not in st.session_state:
    st.session_state.input_list = []

if uploaded_file is not None:
    df = read_csv_file(uploaded_file)
    if df is not None:
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select Column", columns)

        # Additional Input Section
        operator = st.selectbox(
            "Select SQL Operator",
            [
                " ",
                "is equal to",
                "is not equal to",
                "is less than",
                "is less than or equal to",
                "is greater than",
                "is greater than or equal to",
                "is between",
                "is equal to one of the following",
                "is not equal to one of the following",
                "is blank",
                "is not blank",
                "contains",
                "does not contain",
                "contains one of the following",
                "does not contain one of the following",
                "starts with",
                "does not start with",
                "starts with one of the following",
                "does not start with one of the following",
                "ends with",
                "does not end with",
            ],
        )

        # Input text box for query
        additional_value = st.text_input("Additional Value", "")

        # use_date_range = st.checkbox("Include Date Range")
        use_date_range = ""
        if use_date_range:
            additional_date_from = st.date_input(
                "Select Start Date",
                datetime.date.today(),
                key="date_from",
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2023, 12, 31),
            )
            additional_date_to = st.date_input(
                "Select End Date",
                datetime.date.today(),
                key="date_to",
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2023, 12, 31),
            )
        else:
            additional_date_from = None
            additional_date_to = None

        # Create or retrieve word counter instance from session state
        if 'word_counter' not in st.session_state:
            st.session_state.word_counter = WordCounter()
        counter = st.session_state.word_counter

        # Check if the AND or OR button is clicked and save the first input
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            button_clicked = st.button("AND")
        with col2:
            or_button_clicked = st.button("OR")


        if button_clicked:
            first_input = f"{selected_column} {operator} {additional_value} and "
            selected_column = ""
            operator = ""
            additional_value = ""
        elif or_button_clicked:
            first_input = f"{selected_column} {operator} {additional_value} or "
            selected_column = ""
            operator = ""
            additional_value = ""

        # Store the first input in session state
        if first_input:
            st.session_state.input_list.append(first_input)

        # Display the saved inputs with remove option
        st.sidebar.markdown("### Saved Inputs:")
        for i, input_text in enumerate(st.session_state.input_list):
            col1, col2 = st.sidebar.columns([9, 1])
            col1.write(f"{i + 1}. {input_text}")
            if col2.button("❌", key=f"remove_{i}"):
                del st.session_state.input_list[i]

        # Execute SQL query and display result
        if st.button("Run Query"):
            if selected_column and operator or additional_value:
                if additional_date_from and additional_date_to:
                    date_from = additional_date_from.strftime("%m/%d/%Y")
                    date_to = additional_date_to.strftime("%m/%d/%Y")
                    final_input_text = f"{selected_column} {operator} {additional_value} and date is '{date_from}' to '{date_to}'"
                else:
                    final_input_text = f"{selected_column} {operator} {additional_value}"

                # Store the final input in session state
                st.session_state.input_list.append(final_input_text)

                # Combine all inputs from session state
                combined_input = " ".join(st.session_state.input_list)


                # Execute the SQL query
                final_result = get_sql(combined_input, df, counter)

                output_column, stats_column = st.columns(2)
                with output_column:
                    st.title("Output")
                    if isinstance(final_result, list):
                        final_result = pd.DataFrame(final_result)
                    show_data = st.dataframe(
                        final_result.style.set_properties(
                            **{"font-size": "12px", "text-align": "center"}
                        )
                    )
                with stats_column:
                    st.title("Word Count")
                    st.write(f"Total words used: {counter.get_total_words()}")
                    st.write(f"Remaining words: {counter.get_remaining_words()}")

    else:
        st.warning("Please upload a valid CSV file first.")
