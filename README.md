# Flipkart-GRID-5.0
FlipFashion Genie : Your personalized Outfit Wizard

A groundbreaking project at the intersection of Conversational Recommender Systems (CRS), Large Language Models (LLMs), and cutting-edge GenAI technologies.

In this season of Flipkart GRID 5.0, we are entering to elevate the fashion discovery experience by harnessing the power of Generative AI to create a Conversational Fashion Outfit Generator.

With FlipFashion Genie, users will embark on a natural, interactive journey that will empower them to explore fashion choices aligned with their unique style and the latest trends. 

## Demo video link
🟡[Demo Video Link (GoogleDrive)](https://drive.google.com/file/d/1GOG08OZHiCUhxqW914xlhj9qCPoCGE_z/view?usp=sharing)

## Screenshot 

<img src="https://github.com/the-sanyam/Flipkart-GRID-5.0/blob/main/Images/Screenshot%20from%202023-08-20%2011-42-34.png" alt="SS" border="0" >  

## UseCases

With the ability to analyze current fashion trends and incorporate user feedback, our solution will inspire confidence in shoppers' fashion choices while saving them time and decision fatigue.

* Personalization
Our AI analyzes past purchases, browsing, and preferences to create personalized outfit suggestions matching individual taste.

* Friendly Interface
offers a friendly chatbot that interacts naturally, providing outfit suggestions through text, resembling a fashion consultation.

* Special Occasion Styling
For special events like weddings, parties, or festivals, users can engage in a conversation to get outfit recommendations that match the occasion's theme, regional trends, and personal style.

* Trendy Outfit Ideas
Stay updated with the latest fashion trends by conversing with the AI, which combines insights from social media trends with your individual preferences to suggest on-trend outfits.

## Approach

* **Dialogue management module** :- Employs an LLM to engage in conversations with users, maintain contextual understanding, and perform system actions like initiating requests to a recommendation engine. This unifies the entire process as a language modeling task.

* **Recommendation engine** :- A comprehensive conceptual framework is outlined for conducting retrieval using an LLM across an extensive item database. Different strategies are used to meet specific needs based on the data at hand.

* **User profiles and fashion trends** :- Integration of  user profiles, along with real-time fashion trends, as supplementary inputs to system LLMs. This enriches session-level context and enhances the personalized experience.

* **Ranker Module** :- The ranker module leverages an LLM to correlate preferences inferred from user profiles, fashion trends, and ongoing conversations with item details we retrieved previously. This results in a personalized set of recommendations displayed to the user. Additionally, the LLM generates explanations for its decisions, enhancing transparency.

## TechStack

* **OpenAI API**
* **LangChain**
* **Chainlit**
* **Redis**

## Installation


### Pre-Requisites:
1. Install Git Version Control
[ https://git-scm.com/ ]

### Clone the project:

```bash
  git clone https://github.com/st2251/Flipkart-Store-in-MetaVerse.git
```
* Go to the project directory

* Download these into your directory:
```bash
pip install langchain==0.0.123
pip install redis==4.5.3
pip install openai==0.27.2
pip install numpy
pip install pandas             
```

* Now, once you install these requirements, run your project using the below command on different terminals:
```bash
sudo docker run -p 6379:6379 redislabs/redisearch:latest 
chainlit run redis_langchain_chatbot.py -w                   
```
## Documentation

[PPT Documentation](https://drive.google.com/file/d/192TvsyfbdwjmdQRYeYozvTWteCe0S09S/view?usp=sharing)

## Authors

  > [Sanyam Jain](https://github.com/the-sanyam)
  
  > [Aniket Jain](https://github.com/Aniket-Jain-Aman)
  
  > [Rajneesh Kushwaha](https://github.com/Rajneesh2002)
 
#### Made with perseverance and love by 
#### Team CodeTrio ❤️
#### As a solution for Level-2 problem statement given by Flipkart in Flipkart GRID 5.0 Hackathon Challenge




