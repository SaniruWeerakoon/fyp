document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const createdDate = new Date(formData.get("user_created"));
    const now = new Date();
    const accountAgeDays = Math.floor((now - createdDate) / (1000 * 60 * 60 * 24));

    const data = {
        model: formData.get("model"),
        tweet_text: formData.get("tweet_text"),
        timestamp: new Date(formData.get("timestamp")).toISOString(),
        user_followers: parseInt(formData.get("user_followers")),
        user_account_age_days: accountAgeDays,
        user_blue: formData.get("user_blue") === "on",
        hashtags: formData.get("hashtags").split(",").map(tag => tag.trim()).filter(Boolean),
    };

    const resBox = document.getElementById("results");
    resBox.classList.remove("d-none", "alert-danger", "alert-success");
    resBox.classList.add("alert-secondary");
    resBox.innerHTML = "⌛ Predicting...";

    try {
        const res = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await res.json();
        if (res.ok) {
            resBox.className = "alert alert-success mt-4";
            resBox.innerHTML = `
        <strong>Predicted Engagement:</strong><br>
        Views: <b>${result.predicted_views}</b><br>
        Likes: <b>${result.predicted_likes}</b><br>
        Comments: <b>${result.predicted_comments}</b><br>
        Retweets: <b>${result.predicted_retweets}</b><br>
        Quotes: <b>${result.predicted_quotes}</b>
      `;
        } else {
            resBox.className = "alert alert-danger mt-4";
            resBox.innerText = `${result.error}`;
        }
    } catch (err) {
        resBox.className = "alert alert-danger mt-4";
        resBox.innerText = `Request failed: ${err}`;
    }
});
