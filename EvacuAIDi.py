import streamlit as st
import pandas as pd
import networkx as nx
from PIL import Image
from gtts import gTTS
from googletrans import Translator
import time
import glob
import os
import openai
#from dotenv import load_dotenv


#load_dotenv()

# Set the API key from the .env file
#openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = st.secrets["openai_secret_key"]

try:
    os.mkdir("temp")
except:
    pass

translator = Translator()

# Define exits, stairs, and aisles based on your stadium structure
exits = ['EF', 'EG', 'EH', 'ED' , 'EC' , 'EB' , 'EA' , 'EK' , 'EJ']
stair_nodes = [f'S{i}' for i in range(1, 36) if i != 13]  # Excluding S13
aisle_nodes = [f'A{i}' for i in range(1, 30)]  # Assuming aisles are labeled from A1 to A29

# Function to find the nearest node from a given set of nodes
def find_nearest_node(G, start_node, target_nodes):
    nearest_node = None
    nearest_distance = float('inf')
    for node in target_nodes:
        if nx.has_path(G, start_node, node):
            distance = nx.dijkstra_path_length(G, start_node, node, weight='weight')
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node = node
    return nearest_node, round(nearest_distance)

# Function to find the path from a seat to the nearest exit following the stadium's structure
def find_path_seat_to_exit(G, start_seat_node, stair_nodes, aisle_nodes, exits):
    # Find the nearest stair from the seat
    nearest_stair, stair_distance = find_nearest_node(G, start_seat_node, stair_nodes)
    
    # Find the nearest aisle from the stair
    nearest_aisle, aisle_distance = find_nearest_node(G, nearest_stair, aisle_nodes)
    
    # Find the nearest exit from the aisle
    nearest_exit, exit_distance = find_nearest_node(G, nearest_aisle, exits)

    # Combine the paths and distances
    path_seat_to_stair = nx.dijkstra_path(G, start_seat_node, nearest_stair, weight='weight')
    path_stair_to_aisle = nx.dijkstra_path(G, nearest_stair, nearest_aisle, weight='weight')[1:]  # Skip the stair node
    path_aisle_to_exit = nx.dijkstra_path(G, nearest_aisle, nearest_exit, weight='weight')[1:]  # Skip the aisle node
    full_path = path_seat_to_stair + path_stair_to_aisle + path_aisle_to_exit
    full_distance = stair_distance + aisle_distance + exit_distance

    return full_path, round(full_distance), nearest_exit

def calculate_evacuation_time(path, G, pedestrian_type):
    # Define walking speeds in ft/s for different pedestrian types
    walking_speeds = {
        'Non-disabled': {'Aisle': 4.1 , 'Stair': 2.3},
        'Visually impaired': {'Aisle': 2.8, 'Stair': 2},
        'Hearing impaired': {'Aisle': 4.1, 'Stair': 2.3},
        'Low stamina': {'Aisle': 2.6, 'Stair': 1.18}
    }
    
    total_time = 0  # Initialize total time to 0
    
    # Get the walking speeds for the current pedestrian type
    aisle_speed = walking_speeds[pedestrian_type]['Aisle']
    stair_speed = walking_speeds[pedestrian_type]['Stair']
    
    # Iterate through the path and calculate the time for each segment
    for i in range(len(path) - 1):
        segment = G[path[i]][path[i + 1]]
        distance = segment['weight']
        # Check the type of the current node to determine the walking speed
        if 'S' in path[i]:  # Assuming stair nodes contain 'S'
            total_time += distance / stair_speed
        else:  # Else, it is an aisle or exit
            total_time += distance / aisle_speed

    return total_time

# Load nodes and edges from CSV
node_df = pd.read_csv('nodes_capacity.csv')
edges_df = pd.read_csv('edges.csv')

def load_data():
    # Create a graph
    G = nx.Graph()

    # Add nodes with attributes
    for index, row in node_df.iterrows():
        G.add_node(row['Node ID'], label=row['label'])

    # Add edges to the graph with weights converted from inches to feet
    for index, row in edges_df.iterrows():
        G.add_edge(row['Node 1'], row['Node 2'], weight=row['Weight'] / 12)

    return G

# Define a function to perform translation and TTS
def translate_and_text_to_speech(text, output_language):
    translator = Translator()
    translation = translator.translate(text, dest=output_language)
    tts = gTTS(translation.text, lang=output_language)

    # Generate a valid, shorter filename for the audio file
    file_name = f"audio_output_{int(time.time())}.mp3"
    tts.save(file_name)
    return file_name, translation.text


def text_to_speech(text, output_language):
    translation = translator.translate(text, dest=output_language)
    trans_text = translation.text
    tts = gTTS(trans_text, lang=output_language, slow=False)

    # Sanitize file name by removing invalid characters
    sanitized_file_name = text[0:20].replace('"', '').replace('/', '').replace('\\', '')

    # Ensure the 'temp' directory exists
    if not os.path.exists('temp'):
        os.makedirs('temp')

    # Save the file
    file_path = f"temp/{sanitized_file_name}.mp3"
    tts.save(file_path)
    return sanitized_file_name, trans_text

def generate_description_with_gpt(full_path, full_distance, exit_node, pedestrian_type, evacuation_time_minutes):
    # Replace codes in the path with full names
    named_path = []
    for node in full_path:
        if node.startswith('N'):
            named_path.append(f'seat area {node[1:]}')
        elif node.startswith('S'):
            named_path.append(f'stair number {node[1:]}')
        elif node.startswith('A'):
            named_path.append(f'corridor {node[1:]}')
        elif node.startswith('E'):
            named_path.append(f'Exit {node[1:]}')

    # Prepare the prompt for GPT-3.5-turbo
    prompt = (f"Write clear and concise fotball stadium emergency evacuation instructions for a {pedestrian_type} "
          f"from a stadium seat to the nearest exit as a continuous narrative. The path includes {', '.join(named_path)}. "
          f"The total distance to Exit {exit_node[-1]} is {full_distance} feet, "
          f"with an estimated evacuation time of approximately {evacuation_time_minutes} minutes. "
          f"Present the instructions in a flowing paragraph format without bullet points or numbers."
          f"It should be short, concise and just to the point in the emergency evacuation condition. Also, don't say anything about directions like go left, right and etc.")

    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=250
)


    return response.choices[0].text.strip()
def main():
    st.set_page_config(page_title='Stadium Evacuation' , page_icon="🏃‍♂️" )
    st.title("Maverik Stadium Evacuation Path Finder")

    st.write("This app helps you find the quickest evacuation route from your seating area in Maverik Stadium using the Dijkstra algorithm. "
            "Select your current seating area and type of pedestrian to see the best evacuation path.")


    with st.expander("Process and Hypotheses Considered"):
        st.markdown("""
        ## Process and Hypotheses Considered

        #### 1. **Map-Based Node and Edge Extraction:**
        We initiated the process by using a downloadable stadium map. Employing image processing techniques and OpenCV, we extracted x and y coordinates for nodes (seats, exits, stairs) and edges (corridors). This was essential for accurately representing the stadium's structure in our network graph.

        #### 2. **Distance Measurement and Node Capacity Estimation:**
        Real-world distances for the edges were calculated using benchmarks from Google Earth combined with image processing. Node capacities were estimated based on established formulas, ensuring the model reflects real-world conditions within the stadium.

        #### 3. **Evacuation Time Computation:**
        The evacuation times are based on [A comparative study of evacuation strategies for people with disabilities in high-rise building evacuation](https://doi.org/10.1016/j.eswa.2012.07.017). This study provides data on varied walking speeds across different individual types and environments, including stairs.

        #### 4. **Customized Dijkstra Algorithm for Pathfinding:**
        A stadium-specific version of the Dijkstra algorithm was developed to determine the shortest evacuation paths, taking into account the unique structural aspects of the stadium.

        #### 5. **Route Guidance and Multilingual Support:**
        We utilized GPT-3.5-turbo for generating adaptable route choice guidance for various pedestrian types, complemented with voice output and translation capabilities in multiple languages.

        ### Hypotheses and Simplifications

        - **Map-Based Simplifications:** The use of a static image map introduces certain simplifications in representing the dynamic and complex nature of stadium layouts and crowd movements.

        - **Exclusion of Dynamic Crowd Behavior:** Current models do not account for dynamic crowd behavior, which is crucial in real evacuation scenarios. Factors like crowd density, individual movement patterns, and interaction effects are not considered, which are vital in crowd evacuation science.

        - **Emergency Scenarios:** The model does not simulate different emergency scenarios, which can significantly alter evacuation patterns due to panic, reduced visibility, or obstacles.

        - **Limited Accessibility Considerations:** This version does not include specific pathways or accommodations for individuals using wheelchairs, which is a significant aspect of inclusive evacuation planning.

        - **Overlooking Queueing and Bottlenecks:** In real evacuation scenarios, bottlenecks and queueing can occur at exits or narrow pathways, significantly impacting evacuation times. These aspects are not yet incorporated into our model.

        - **Assumption of Constant Walking Speeds:** The model assumes constant walking speeds as per the referenced study, not accounting for variations due to crowd density, individual fitness, or psychological factors in emergencies.

        These hypotheses and simplifications are specific to the current version of our model. They will be considered and addressed step by step in upcoming versions, as we aim to enhance the accuracy, inclusivity, and realism of our evacuation planning tool.
        """)


    st.header("Maverik Stadium Map")
    map_image = Image.open("Maverik Stadium Map.jpeg")
    st.image(map_image)

    with st.expander("Stadium Network"):
        network_image = Image.open("network.png")
        st.image(network_image)

    
    G = load_data()

    # Assuming 'label' column indicates the type of each node and 'Node ID' contains the node names
    seat_nodes = node_df[node_df['label'] == 'Seat']['Node ID'].tolist()  # Filter nodes labeled as seats
    seat_node_options = [node.replace('N', '') for node in seat_nodes]  # Remove 'N' from the node names

    start_node = st.selectbox("Choose seating area (node):", seat_node_options, key="seat_selection")


    pedestrian_types = ['Non-disabled', 'Visually impaired', 'Hearing impaired', 'Low stamina']
    pedestrian_type = st.selectbox("Choose pedestrian type:", pedestrian_types)

    # Get the user's choice and prepend 'N'
    selected_seat = 'N' + start_node


    # Language selection for output
    out_lang = st.selectbox(
    "Select output language",
    ("English", "Chinese", "French", "Hindi", "korean", "Japanese"),
    )
    if out_lang == "English":
        output_language = "en"
    elif out_lang == "Hindi":
        output_language = "hi"
    elif out_lang == "French":
        output_language = "fr"
    elif out_lang == "korean":
        output_language = "ko"
    elif out_lang == "Chinese":
        output_language = "zh-cn"
    elif out_lang == "Japanese":
        output_language = "ja"

    if st.button("Find Path"):
        if selected_seat in G:
            # Finding the path from the seat to the nearest exit
            full_path, full_distance, exit_node = find_path_seat_to_exit(G, selected_seat, stair_nodes, aisle_nodes, exits)
        
            # Calculating evacuation time for the selected pedestrian type
            evacuation_time = calculate_evacuation_time(full_path, G, pedestrian_type)
            evacuation_time_minutes = round(evacuation_time / 60, 0)

            # Now pass the evacuation_time_minutes to the function
            description = generate_description_with_gpt(full_path, full_distance, exit_node, pedestrian_type, evacuation_time_minutes)

            st.write(description)

            # Translate and convert to speech
            result_file, translated_text = text_to_speech(description, output_language)

            if output_language != "en":
                st.write(translated_text)

            # Open the audio file and play it
            audio_file = open(f"temp/{result_file}.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3", start_time=0)

        else:
            st.error(f"Starting node {selected_seat} not found in the graph.")


    def remove_files(n):
        mp3_files = glob.glob("temp/*mp3")
        if len(mp3_files) != 0:
            now = time.time()
            n_days = n * 86400
            for f in mp3_files:
                if os.stat(f).st_mtime < now - n_days:
                    os.remove(f)
                    print("Deleted ", f)


    remove_files(7)
    
    with st.expander("Future Tasks and Enhancements"):
        st.markdown("Planned Improvements for Upcoming Versions")

        # Future tasks with disabled checkboxes
        st.checkbox("Consider Map-Based Simplifications and Dynamic Crowd Behavior", disabled=True)
        st.checkbox("Simulate Emergency Scenarios and Their Impact on Evacuation", disabled=True)
        st.checkbox("Incorporate Pathways and Accommodations for Wheelchair Users", disabled=True)
        st.checkbox("Account for Queueing and Bottlenecks in Evacuation Scenarios", disabled=True)
        st.checkbox("Adjust for Variations in Walking Speeds due to Various Factors", disabled=True)
        st.checkbox("Implement Agent-Based Simulation and Reinforcement Learning for Enhanced Wayfinding", disabled=True)
        st.checkbox("Utilize More Naturalistic Voice in Route Guidance", disabled=True)
        st.checkbox("Tune the Language Model to Produce Consistent and Accurate Guidance", disabled=True)
        st.checkbox("Integrate a Chatbot System for Interactive Guidance and Support", disabled=True)
        st.checkbox("Incorporate Advanced Map Navigation Features", disabled=True)
        st.checkbox("Leverage Building Information Modeling (BIM) for Detailed Spatial Analysis", disabled=True)
        st.checkbox("Consider Detailed Pedestrian Behavior Models in Simulations", disabled=True)
        st.checkbox("Account for Structural Features in Evacuation Route Planning", disabled=True)
        st.checkbox("Develop Specialized Route Choice Models for Disabled Individuals", disabled=True)

if __name__ == "__main__":
    main()
