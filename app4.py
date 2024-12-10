import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gdown


@st.cache_data
def download_and_load_model():
    file_id = "1Oip1pPt4DF5Yy6z8V4G6iXA4-uRLjqhb"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "random_forest.pkl"

    # Download the model file
    gdown.download(url, output, quiet=False)

    # Load the model
    import pickle
    with open(output, "rb") as f:
        model = pickle.load(f)

    return model

# Load the model
model = download_and_load_model()

# Load data
@st.cache_data
def load_data():
    user_profiles = pd.read_csv('user_profiles.csv')
    item_profiles = pd.read_csv('item_profiles.csv')
    final_combined_df1 = pd.read_csv('final_combined_df1.csv')
    return user_profiles, item_profiles, final_combined_df1

# Load datasets
user_profiles, item_profiles, final_combined_df1 = load_data()

# Create a mapping of Item ID_tourist to Item Name and Category
item_name_mapping = final_combined_df1.set_index('Item ID_tourist')['Item_name'].to_dict()

# Category mapping
category_mapping = {
    "12 Springs of Prophet Moses (A.S.) - Tabuk Province": "Natural Landscapes",
    "A Trip To The Edge of The World - Riyadh Province": "Natural Landscapes",
    "Abha High City - Asir Province": "Natural Landscapes",
    "Abraj Al Bait Towers - The Clock Towers Complex - Makkah": "Religious Sites",
    "Abu Kheyal Park - Beautiful Hill Top Park in Abha": "Natural Landscapes",
    "Al Bujairi Heritage Park - Riyadh": "Cultural & Heritage Sites",
    "Al Faqir Well - Madinah": "Religious Sites",
    "Al Ghamama Mosque - Madinah": "Religious Sites",
    "Al Ghufran Safwah Hotel Makkah": "Hotels & Accommodations",
    "Al Hair (Ha-ir) Parks and Lakes - Riyadh": "Natural Landscapes",
    "Al Jawatha Historic Mosque - Hofuf - Eastern Province": "Religious Sites",
    "Al Madinah Museum - Hejaz Railway Railway Museum": "Historical Sites",
    "Al Marwa Rayhaan Hotel by Rotana": "Hotels & Accommodations",
    "Al Masjid Al Haram - Makkah": "Religious Sites",
    "Al Masmak Palace Museum": "Historical Sites",
    "Al Qarah Mountain - The Land of Civilisation - Al Ahsa": "Natural Landscapes",
    "Al Rahma - Floating Mosque - Jeddah": "Religious Sites",
    "Al Safwah Royale Orchid Hotel": "Hotels & Accommodations",
    "Al Sahab Park - Asir Province": "Natural Landscapes",
    "Al Shafa Mountain Park - Taif": "Natural Landscapes",
    "Al Shohada Hotel": "Hotels & Accommodations",
    "Al Soudah View Point - Asir Province": "Natural Landscapes",
    "Al Tawba Mosque - Tabuk Province": "Religious Sites",
    "Al Ula - Old Town - Madinah Province": "Cultural & Heritage Sites",
    "Al Wabhah Crater - Makkah Province": "Natural Landscapes",
    "Anjum Hotel Makkah": "Hotels & Accommodations",
    "Anwar Al Madinah Movenpick Hotel": "Hotels & Accommodations",
    "Arch (Rainbow) Rock - Al Ula": "Natural Landscapes",
    "As Safiyyah Museum and Park - Madinah": "Cultural & Heritage Sites",
    "Asfan Fortress - Makkah Province": "Historical Sites",
    "Ash Shafa View Point - Taif": "Natural Landscapes",
    "Baab Makkah Jeddah - Makkah Gate": "Historical Sites",
    "Bir Al Shifa Well - Madinah Province": "Religious Sites",
    "Boulevard - Luxury Shopping Mall - Jeddah": "Shopping & Entertainment",
    "Boulevard Riyadh City - Riyadh": "Shopping & Entertainment",
    "Boulevard World - Riyadh": "Shopping & Entertainment",
    "Catalina Seaplane Wreckage - Tabuk Province": "Historical Sites",
    "Cave at Al Qarah (Jabal) Mountain - Al Ahsa": "Natural Landscapes",
    "Clock Tower Museum - Makkah": "Historical Sites",
    "Cornish Al Majidiyah - Al Qatif - Eastern Province": "Beaches & Waterfronts",
    "Dallah Taibah Hotel": "Hotels & Accommodations",
    "Dammam Corniche - Eastern Province": "Beaches & Waterfronts",
    "Dar Al Madinah Museum - Madinah": "Historical Sites",
    "Desert Ruins of Al Ula - Saudi's Forgotten Past": "Historical Sites",
    "Discover Jabal Ithlib - Archeological Site - Al Ula": "Cultural & Heritage Sites",
    "Discover Jeddah Waterfront on The Red Sea": "Beaches & Waterfronts",
    "Dorrar Aleiman Royal Hotel (Dar Al Eiman Royal Hotel)": "Hotels & Accommodations",
    "Dumat Al Jandal Lake - Al Jawf Province": "Natural Landscapes",
    "Elaf Al Mashaer Hotel Makkah": "Hotels & Accommodations",
    "Elaf Kinda Hotel": "Hotels & Accommodations",
    "Emaar Royal Hotel Al Madinah": "Hotels & Accommodations",
    "Explore New Jeddah Corniche on The Red Sea": "Beaches & Waterfronts",
    "Explore Old Jeddah, Al Balad": "Cultural & Heritage Sites",
    "Fakieh Aquarium - Jeddah": "Modern Attractions",
    "Fanateer Beach - Jubail - Eastern Province": "Beaches & Waterfronts",
    "Four Points by Sheraton Makkah Al Naseem": "Hotels & Accommodations",
    "Harrat Kishb Volcanic Field": "Natural Landscapes",
    "Hegra (Al Hijr) - Mada'in Saleh - Al Ula": "Cultural & Heritage Sites",
    "Hilton Makkah Convention Hotel": "Hotels & Accommodations",
    "Historical At-Turaif World Heritage Site - Riyadh": "Historical Sites",
    "Historical Diriyah Museum - Riyadh": "Historical Sites",
    "Al Madinah Museum - Hejaz Railway Railway Museum": "Historical Sites",
    "Hussainiya Park - Makkah": "Natural Landscapes",
    "Ikmah Heritage Mountain - Jabal Ikmah - Al Ula": "Cultural & Heritage Sites",
    "Jabal Al Nour (Noor) - Mount of Revelation - Makkah": "Religious Sites",
    "Jabal Dakka Park - View Point - Taif": "Natural Landscapes",
    "Jamarat Bridge - Makkah": "Religious Sites",
    "Jannat Al-Mu'alla Cemetry - Makkah": "Religious Sites",
    "Jannatul Baqi Cemetry - Madina": "Religious Sites",
    "Jawatha Park - Hofuf - Eastern Province": "Natural Landscapes",
    "Jeddah Sign": "Modern Attractions",
    "Jeddah Yacht Club and Marina": "Modern Attractions",
    "Khaybar Historical City - Madinah Province": "Historical Sites",
    "Khobar Corniche - Eastern Province": "Beaches & Waterfronts",
    "Khobar Corniche Mosque - Eastern Province": "Religious Sites",
    "King Abdulaziz Center for World Culture - Ithra - Dharan": "Cultural & Heritage Sites",
    "King Abdulaziz Historical Center - National Museum": "Historical Sites",
    "King Abdullah Financial District - KAFD - Riyadh": "Modern Attractions",
    "King Fahad Causeway - Eastern Province": "Modern Attractions",
    "King Fahad's Fountain - Jeddah": "Modern Attractions",
    "King Fahd Glorious Quran Printing Complex - Madinah": "Religious Sites",
    "Kingdom Arena - Riyadh": "Modern Attractions",
    "Kingdom Centre Tower - Mall & Hotel - Riyadh": "Modern Attractions",
    "Lake Park Namar Dam - Riyadh": "Natural Landscapes",
    "M Hotel Makkah Millennium": "Hotels & Accommodations",
    "Makkah Clock Royal Tower, A Fairmont Hotel": "Hotels & Accommodations",
    "Makkah Hotel": "Hotels & Accommodations",
    "Makkah Towers": "Hotels & Accommodations",
    "Marid Castle - Al Jawf Province": "Historical Sites",
    "Masjid Addas - Taif": "Religious Sites",
    "Masjid Al Ijabah - Madinah": "Religious Sites",
    "Masjid Al Jinn - Makkah": "Religious Sites",
    "Masjid Al Qiblatayn - Madinah": "Religious Sites",
    "Masjid Bilal (RA) - Madinah": "Religious Sites",
    "Masjid Imam Bukhari - Madinah": "Religious Sites",
    "Masjid Miqat (Dhul Hulaifah) - Madinah": "Religious Sites",
    "Masjid Quba - Madinah": "Religious Sites",
    "Masjid Sajdah (Abu Dhar Al Ghifari) - Madinah": "Religious Sites",
    "Masjid Shuhada Uhud - Madinah": "Religious Sites",
    "Matbouli House Museum - Jeddah": "Historical Sites",
    "Millennium Makkah Al Naseem Hotel": "Hotels & Accommodations",
    "Mount Arafat - Makkah": "Religious Sites",
    "Mount Uhud - Archers' Hill - Madinah": "Natural Landscapes",
    "Movenpick Hotel & Residence Hajar Tower Makkah": "Hotels & Accommodations",
    "Murabba Historical Palace - Riyadh": "Historical Sites",
    "Murjan Island - Eastern Province": "Beaches & Waterfronts",
    "Nassif House Museum - Jeddah": "Historical Sites",
    "Oud square": "Shopping & Entertainment",
    "Pullman Zamzam Madina": "Hotels & Accommodations",
    "Qanona Valley - Al Bahah Province": "Natural Landscapes",
    "Qantara Mosque (Masjid Madhoon) - Taif": "Religious Sites",
    "Qasab Salt Flats": "Natural Landscapes",
    "Quba Square - Madinah": "Religious Sites",
    "Ras Tanura Beach - Eastern Province": "Beaches & Waterfronts",
    "Red Sand Dunes - Riyadh Province": "Natural Landscapes",
    "Saudi Aramco Exhibit - Al Dhahran": "Historical Sites",
    "Seven Mosques - Madinah": "Religious Sites",
    "Skybridge at Kingdom Centre - Riyadh": "Modern Attractions",
    "Tabuk Castle (Tabuk Fort)": "Historical Sites",
    "Tayma Fort - Tabuk Province": "Historical Sites",
    "The Ain Ancient Village - Al Bahah Province": "Historical Sites",
    "The Ancient City of Madyan - Tabuk Province": "Historical Sites",
    "The Blessed Al Masjid An Nabawi - The Prophet's Mosque (SAWS)": "Religious Sites",
    "The Farasan Islands": "Natural Landscapes",
    "Umluj - Turquoise Waters of the Red Sea": "Natural Landscapes",
    "VIA Riyadh - Riyadh": "Shopping & Entertainment",
    "Yanbu Waterfront - Madinah Province": "Beaches & Waterfronts",
    "Zamzam Well - Makkah": "Religious Sites",
    "ZamZam Pullman Makkah Hotel": "Hotels & Accommodations",
    "Wadi Tayyib Al Ism - Tabuk Province": "Natural Landscapes",
    "Wadi Namer Waterfall - Riyadh": "Natural Landscapes",
    "Wadi Lajab - Jazan Province": "Natural Landscapes",
    "Wadi Laban Dam": "Natural Landscapes",
    "Al Jawatha Historic Mosque - Hofuf - Eastern Province": "Religious Sites",
    "Wadi Disah - Tabuk Province": "Natural Landscapes",
    "Wadi Dharak Lake Park - Al Bahah Province": "Natural Landscapes",
    "Wadi Al Jinn (Baida) - Madinah Province": "Natural Landscapes",
    "Ushaiqer Heritage Village - Riyadh Province": "Cultural & Heritage Sites",
    "Umm Senman Mountain - Hail Province": "Natural Landscapes",
    "Tomb of Lihyan son of Kuza - Al Ula": "Historical Sites",
    "The Spectacular Elephant Rock - Al Ula": "Natural Landscapes",
    "The Seven Mosques (Saba Masajid) - Madinah": "Religious Sites",
    "The Makkah (Mecca) Museum": "Cultural & Heritage Sites",
    "The Ghars Well - Madinah - بئر غرس": "Natural Landscapes",
    "The Garden of Salman Al Farsi (RA) - Madinah": "Cultural & Heritage Sites",
    "The Ethiq Well and Garden - Madinah": "Cultural & Heritage Sites",
    "The Art Street - Abha": "Modern Attractions",
    "Tarout Island - Eastern Province": "Historical Sites",
    "Taif Cable Cars (Telefric Al Hada)": "Modern Attractions",
    "Taiba Hotel Madinah": "Hotels & Accommodations",
    "Tabuk Ottoman Castle": "Historical Sites",
    "Jabbal Dakka Park": "Natural Landscapes",
    "Swissotel Makkah": "Hotels & Accommodations",
    "Swissotel Al Maqam Makkah": "Hotels & Accommodations",
    "Sparhawk Arch -Rock Formation - Al Ula": "Historical Sites",
    "Sofitel Shahd al Madinah": "Hotels & Accommodations",
    "Sky Bridge Kingdom Tower": "Modern Attractions",
    "Saqifah Bani Saidah - Madinah": "Cultural & Heritage Sites",
    "Riyadh Zoo Day Trip - Riyadh Province": "Modern Attractions",
    "Rijal Almaa - Beautiful Heritage Village": "Cultural & Heritage Sites",
    "Quba Walkway Park- Madinah": "Modern Attractions",
    "Quba Square - Madinah - ساحة قباء": "Modern Attractions",
    "Oud square - عُود سكوير": "Modern Attractions",
    "Nassif House Museum -Jeddah": "Cultural & Heritage Sites",
    "Lake Park Namar Dam -Riyadh": "Natural Landscapes",
    "King Abdullah Financial District  - KAFD - Riyadh": "Modern Attractions",
    "Jabal Dakka Park  - Taif - جبل دكا": "Natural Landscapes",
    "Bir Al Shifa Well - Madinah Province - بئر الشفاء": "Cultural & Heritage Sites",
    "Al Madinah Museum - Hejaz Railway Railway Museum - متحف السكة الحديد": "Cultural & Heritage Sites"
}

# App Title
st.title("Personalized Tourist Experience Recommender")
st.write("Explore and get recommendations for the best attractions in Saudi Arabia!")

# Define Tabs
tabs = st.tabs(["Home", "Recommendation Engine", "Best Attractions", "Login"])

# Home Tab
with tabs[0]:
    st.header("Home")
    st.write("Welcome to the Tourist Experience Recommender!")
    st.write("Use this app to explore attractions and get personalized recommendations.")
    st.image("backgroundleft.jpg", caption="Discover Saudi Arabia's Best Attractions", use_container_width=True)

# Recommendation Engine Tab
with tabs[1]:
    st.header("Recommendation Engine")
    st.write("Get tailored recommendations based on your preferences!")

    st.write("### Enter User ID")
    user_id = st.text_input("Enter User ID:")

    if user_id:
        user_id = user_id.strip().lower()
        if user_id in user_profiles['User ID'].str.lower().values:
            # Fetch user profile
            user_profile = user_profiles[user_profiles['User ID'].str.lower() == user_id].iloc[0]

            # Generate dummy SVD scores for illustration
            svd_scores = np.random.rand(len(item_profiles))

            # Calculate cosine similarity scores
            user_features = np.array([user_profile['activity_level']] * 2).reshape(1, -1)
            item_features = item_profiles[['City_Sentiment_Score', 'Avg_City_Sentiment_Score']].values
            kbf_scores = cosine_similarity(user_features, item_features).flatten()

            # Combine SVD and KBF scores
            hybrid_scores = 0.6 * svd_scores + 0.4 * kbf_scores
            item_profiles['Hybrid Score'] = hybrid_scores.round(4)

            # Filters for Province and Category
            province_filter = st.selectbox(
                "Filter by Province:",
                options=['All'] + list(item_profiles['Province'].unique())
            )
            category_filter = st.selectbox(
                "Filter by Category:",
                options=['All'] + list(set(category_mapping.values()))
            )

            filtered_recommendations = item_profiles.copy()
            if province_filter != 'All':
                filtered_recommendations = filtered_recommendations[filtered_recommendations['Province'] == province_filter]

            # Map Item ID to Item Name and Category
            filtered_recommendations['Item_name'] = filtered_recommendations['Item ID_tourist'].map(item_name_mapping)
            filtered_recommendations['Item Category'] = filtered_recommendations['Item_name'].map(category_mapping)

            if category_filter != 'All':
                filtered_recommendations = filtered_recommendations[filtered_recommendations['Item Category'] == category_filter]

            # Sort and select top recommendations
            recommendations = filtered_recommendations.sort_values(by='Hybrid Score', ascending=False).head(5)

            if recommendations.empty:
                st.warning("No recommendations found for the selected filters.")
            else:
                # Display recommendations
                recommendations['Rank'] = range(1, len(recommendations) + 1)
                recommendations = recommendations[['Rank', 'Item_name', 'City', 'Item Category', 'Hybrid Score']]
                st.markdown(recommendations.to_markdown(index=False))
        else:
            st.error("User ID not found! Please create one in the Login tab.")

# Best Attractions Tab
with tabs[2]:
    st.header("Best Attractions")
    st.write("Explore the top attractions based on popular choices!")

    # Generate dummy scores for best attractions
    item_profiles['Popularity Score'] = np.random.rand(len(item_profiles))
    best_attractions = item_profiles.sort_values(by='Popularity Score', ascending=False).head(10)

    # Map Item ID to Item Name and Category
    best_attractions['Item_name'] = best_attractions['Item ID_tourist'].map(item_name_mapping)
    best_attractions['Item Category'] = best_attractions['Item_name'].map(category_mapping)

    # Display the best attractions
    for index, row in best_attractions.iterrows():
        st.markdown(
            f"""
            <div style="border:1px solid #ddd; padding:10px; margin-bottom:10px; border-radius:5px;">
                <h4>{row['Item_name']}</h4>
                <p><strong>City:</strong> {row['City']}</p>
                <p><strong>Category:</strong> {row['Item Category']}</p>
                <p><strong>Popularity Score:</strong> {row['Popularity Score']:.2f}</p>
            </div>
            """, unsafe_allow_html=True
        )

# Login Tab
with tabs[3]:
    st.header("Login")
    st.write("Create a new account or log in to access personalized recommendations.")

    new_user_id = st.text_input("New User ID:")
    preferred_province = st.selectbox("Preferred Province:", ['All'] + list(item_profiles['Province'].unique()))
    category_interest = st.selectbox("Category of Interest:", ['All'] + list(set(category_mapping.values())))
    activity_level = st.slider("Activity Level (1 to 5):", min_value=1, max_value=5, value=3)

    if st.button("Create Account"):
        if new_user_id.strip().lower() in user_profiles['User ID'].str.lower().values:
            st.error("User ID already exists!")
        else:
            st.success(f"User '{new_user_id}' created successfully!")