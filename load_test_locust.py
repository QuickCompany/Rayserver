from locust import HttpUser,TaskSet,task,between


class LayoutProcessingTest(TaskSet):
    @task(1)
    def send_pdf_req(self):
        res = self.client.post("/patent",json={
            "slug": "floating-bilayer-tablet-ruits",
            "link": "https://blr1.vultrobjects.com/patents/document/184426534/azure_file/b4c7d47cdeffcf2cc6f0873879527f80.pdf"
        })



class WebsiteUser(HttpUser):
    """
    User class that does requests to the locust web server running on localhost
    """

    host = "http://45.76.165.126:8000"
    wait_time = between(2,4)
    tasks = [LayoutProcessingTest]


