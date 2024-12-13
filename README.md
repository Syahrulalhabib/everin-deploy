# Deploy Model With Cloud Run

**Overview**

This repository is a write up of how to deploy a Machine Learning Model using Cloud Run

**Activate the cloud shell**
**and clone this repository**
```
git clone https://github.com/Syahrulalhabib/everin-deploy.git
```
**Go to the app directory**
```
cd everin-deploy/app
```
**Deployment** 
```
gcloud run deploy everin-deploy --port 8080 --source . --region asia-southeast2 --memory 2Gi
```

**API Endpoint**
========================================
**1. Image Recognition**

POST 
```
https://<service-name>-<random-hash>-<region>.run.app/predict
```

**2. Nutrition Calculator**

POST
```
https://<service-name>-<random-hash>-<region>.run.app/calculate
```

**3. Food Recommendation**

POST
```
https://<service-name>-<random-hash>-<region>.run.app/recommend
```

**4. Recommendation by Name**

POST
```
https://<service-name>-<random-hash>-<region>.run.app/recommend-by-name
```


**5. Submit Data**

POST
```
https://<service-name>-<random-hash>-<region>.run.app/submit
```


**6. Status**

GET 
```
https://<service-name>-<random-hash>-<region>.run.app/status/<email>
```
