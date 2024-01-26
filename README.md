# Stadium Evacuation Path Finder - Streamlit App

## Overview
This repository contains a Streamlit application designed for efficient evacuation planning in stadiums. It leverages advanced algorithms, image processing, and AI to provide personalized evacuation routes based on user location and specific requirements.

## Features
- **Interactive Mapping:** Utilizes Dijkstra's algorithm for optimal evacuation routes.
- **Node and Edge Extraction:** Processes stadium maps to extract key points and paths.
- **Evacuation Time Estimation:** Calculates times based on pedestrian types and walking speeds.
- **Multilingual Voice Guidance:** Offers text-to-speech conversion and supports multiple languages.
- **AI-Generated Route Descriptions:** Uses OpenAI's GPT-3.5-turbo for tailored evacuation instructions.

## Technologies
- **Streamlit**
- **Pandas & NetworkX**
- **Pillow (PIL Fork)**
- **gTTS & googletrans**
- **OpenAI API**
- **Python-Dotenv**

## How to Run
1. Clone the repository:
git clone https://github.com/your-username/your-repository.git

2. Install dependencies:
pip install -r requirements.txt

3. Run the application:
streamlit run app.py

## Future Enhancements
- [ ] Agent-based simulation for realistic wayfinding.
- [ ] More naturalistic voice synthesis.
- [ ] Expansion of the chatbot system.
- [ ] Advanced map navigation features.
- [ ] Incorporation of Building Information Modeling (BIM).
- [ ] Pedestrian behavior models in simulations.

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](link-to-issues-page) for open issues or open a new issue.

## License
Distributed under the MIT License. See `LICENSE` for more information.
