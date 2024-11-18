from app import app

def test_home():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"<title>My Portfolio</title>" in response.data  # Check for the title tag in the HTML
