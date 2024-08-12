# Proiect---Python
Datele au fost preluate de pe Kaggle ( Breast Cancer Wisconsin (Diagnostic) Data Set | Kaggle ) si sunt referitoare la diagnosticul cancerului la san. Atributele setului de date sunt reprezentate de id, de diagnostic (cancer malign sau cancer benign) si caracteristici cu valoare reala calculate pentru fiecare nucleu celular. 

Variabila tinta pentru acest set de date are doua categorii: M, daca cancerul este de tip malign si B daca cancerul este de tip benign. 

Pentru acest proiect au fost utilizate urmatoarele fisiere de intrare:  
“cancer_set1.csv” => contine datele pentru antrenarea si testarea modelului 
“cancer_set2.csv” => continue datele pe care se vor face predictii prin intermediul modelului ales.  

Scorurile discriminante sunt salvate in fisierul “Predictori.csv”
Matricea de confuzie se regaseste in fisierul “MatC_LDA.csv”
Indicatorii cu privire la acuratete sunt salvati in fisierul “Acuratete_LDA.csv” 
Matricea de confuzie se regaseste in fisierul “MatC_B.csv”
Indicatorii cu privire la acuratete sunt salvati in fisierul “Acuratete_B.csv” 


Modelul pe care am ales sa-l utilizez este cel bazat pe analiza liniara discriminanta. 
Cu o diferenta foarte mica intre valori, modelul Bayes are o acuratete mai scazuta decat modelul ales.  
Predictiile realizate au fost salvate in fisierul “Predictii.csv”. 
