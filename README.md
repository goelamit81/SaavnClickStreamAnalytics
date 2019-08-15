# SaavnClickStreamAnalytics

Suppose you are a Big Data engineer at Saavn working closely with the company’s Machine Learning team. For better user engagement, you’re required to build a system that keeps the users notified about new releases based on their music preferences. Suppose, a new track of some particular artist has been released. Your system should push notification about this song to the appropriate set of audience. For example, if Baadshah’s new track “Tareefan” is released, you would probably like to send notifications about this song to the users who prefer to listen to singers like Honey Singh and Raftaar than to users who prefer listening to singers like Jagjit Singh. Pushing a ‘rap song’ notification to an admirer of classical music is not recommended. The user may get annoyed at some time and may even uninstall the app.

To avoid any such repercussions, it is quintessential to push a song notification only to its interested and relevant audience, which in this case will be youngsters interested in rap music. To accommodate this feature in the platform, you need to segregate the users into different cohorts based on some common characteristics and push the notification about any new song to an appropriate user cohort.

 

User Cohorts are a group of users who share common characteristics. Users of the same gender, age group and locality are typical examples of user cohorts. User cohorts also help to segment the user base into meaningful clusters for analytics, marketing campaigns, targeting ads and so on. Creating a cohort based on features such as gender is pretty trivial. But, a majority of Saavn's users do not have a personal account on the platform. Under such circumstances, personalised information such as age, gender, geographical location etc. is unavailable to the developers trying to build a machine learning solution.

 

However, you can still segment the users based on their activity on the platform. In other words, you can create user cohorts using clickstream data. 
