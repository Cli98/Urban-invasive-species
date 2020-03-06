# Urban-invasive-species
Code for Detecting plant invasion in urban parks with aerial image time series and residual neural network.


Background:
The study focuses on detecting a major understory exotic plant, autumn olive (Elaeagnus umbellate) in all the 26 nature preserves (approximately 350,000 acres in total) in Charlotte, North Carolina, USA. Data is confidential and thus this repo cannot provide any data. This is a classification task and we need to know if a provided sample is "invasive" or "non-invasive".

Implementation:
In this project, three challenges are proposed as,
1. How will different years affect invasion process?
2. What's the plant condition for different regions?
3. What's the classification performance with cross-validation?

To deal with those challenges, this repo provides three data loader for each scenarios. Then data is prepared and sent to pytroch deep learning model. Results are reported after training phase concludes.

Please feel free to leave any comments and feedback. Thank you for your interest.
