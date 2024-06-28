# Stock-Price-Analyzer

This project endeavors to create a robust stock price prediction system, leveraging deep learning models and real-time data sources to empower investors, analysts, and researchers in their decision-making processes. The system ensures seamless functionality with a comprehensive four-layer architecture comprising data preprocessing, predictive modeling, sentiment analysis, and an intuitive web interface. Powered by PyTorch for model development, Streamlit for the interface, and real-time APIs for data retrieval, our implementation prioritizes iterative prototyping to streamline development timelines. Future enhancements will focus on integrating more data sources, fine-tuning model parameters, and enriching the web interface for enhanced user experience.

To use the project,
Execute app.py file in the system having python and pytorch environment.

User Class:
The Stock Price Prediction Application caters to a diverse audience, including individual investors, analysts, and researchers, who seek reliable and actionable insights and informed investments. The main user class will be around individual investors with the following characteristics,

•	Characteristics: 
Individual investors are typically retail investors who make investment decisions on their behalf. They may have varying levels of experience in financial markets, ranging from novice investors to seasoned traders.
•	Needs and Expectations: 
Individual investors seek reliable and actionable insights to make informed investment decisions. They require user-friendly interfaces that provide clear and concise information about stock predictions, market trends, and market sentiments. Accessibility and ease of use are essential for this user class, as they may not have specialized knowledge in finance or data science.

Flowchart of the system:

            ![image](https://github.com/Meet1590/Stock-Price-Analyzer/assets/63838050/b411e78e-def7-4116-8148-ca8a19663cef)
                                              (Fig 3: Flow chart of the system)
The proposed system will incorporate the same workflow for LSTM as illustrated in the above diagram. In this flow, various data cleaning and processing techniques are applied such as removing null values, handling duplicate values, etc. Sequentially processed data will be split into training, testing, and validation. LSTM is known for its feature of countering the vanishing gradient problem so this can be done by the reinforcement phase where the model’s output is given as input to the same module repeatedly so that the model can learn.

The architecture of the system:
The architecture of the system is inspired by the four-layer design proposed in a FinGPT framework which is as follows,

                            ![image](https://github.com/Meet1590/Stock-Price-Analyzer/assets/63838050/3ad62f3c-87b2-4c49-9091-415a828c6f9f)
 
                                         (FinGPT Framework,  Yang et al. (2023))

Data Sources
            ![Screenshot (296)](https://github.com/Meet1590/Stock-Price-Analyzer/assets/63838050/ede65733-5a1a-4513-b120-674d498d3730)

UI
            ![image](https://github.com/Meet1590/Stock-Price-Analyzer/assets/63838050/5e817390-176f-4918-8928-81e50672658f)

