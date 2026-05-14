# Unemploymentand Trade Networks: Factor Network Vector Autoregression

Authors: Giacomo Rinaldi · Matteo Severi · Gianluca Antonio Spennacchio

Course: Machine Learning for Economists — University of Bologna

Standard macroeconomic models often ignore the network structure of international trade, risking a misrepresentation of how shocks propagate across countries. This project applies the Factor Network Autoregression (FNAR) of Barigozzi et al. (2025) to address this: latent trade factors are extracted from a Country × Country × Sector × Time tensor built from WIOT data (41 countries, 45 sectors, 2000–2014) via tensor-based PCA, then embedded in a VAR(1) to forecast first-differenced unemployment rates.

Three network factors are retained, capturing globalisation intensity, manufacturing-vs-services specialisation, and essential goods trade. Estimated on the full sample, all three enter significantly — the globalisation and essential-goods factors dampen unemployment propagation, while the services factor amplifies it. Crucially, splitting the sample around 2008 reveals a structural break: network factors are statistically insignificant before the crisis and strongly significant after, indicating that the Financial and Sovereign Debt crises activated trade networks as a key channel for transmitting labour market shocks.
