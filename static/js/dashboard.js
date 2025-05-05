document.addEventListener("DOMContentLoaded", function () {
    const profileBtn = document.getElementById("profileBtn");
    const modal = document.getElementById("profileModal");
    const closeBtn = document.querySelector(".close");
    const profileForm = document.getElementById("profileForm");

    // Open Profile Modal
    profileBtn.addEventListener("click", function () {
        fetchProfileData();
        modal.style.display = "flex";
    });

    // Close Modal
    closeBtn.addEventListener("click", function () {
        modal.style.display = "none";
    });

    // Save Profile Data
    profileForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const userData = {
            height: document.getElementById("height").value,
            weight: document.getElementById("weight").value,
            bloodGroup: document.getElementById("bloodGroup").value,
            medicalHistory: document.getElementById("medicalHistory").value,
        };

        const username = localStorage.getItem("username");

        if (username) {
            fetch(`http://localhost:5000/saveProfile/${username}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(userData),
            })
                .then((res) => res.json())
                .then((data) => {
                    alert("Profile Updated Successfully!");
                    modal.style.display = "none";
                })
                .catch((err) => console.error("Error:", err));
        } else {
            alert("Please log in first.");
        }
    });

    // Fetch Profile Data
    function fetchProfileData() {
        const username = localStorage.getItem("username");

        if (username) {
            fetch(`http://localhost:5000/getProfile/${username}`)
                .then((res) => res.json())
                .then((data) => {
                    if (data) {
                        document.getElementById("height").value = data.height || "";
                        document.getElementById("weight").value = data.weight || "";
                        document.getElementById("bloodGroup").value = data.bloodGroup || "A+";
                        document.getElementById("medicalHistory").value = data.medicalHistory || "";
                    }
                })
                .catch((err) => console.error("Error:", err));
        }
    }

    // // Dynamic Greeting
    // function getGreeting() {
    //     const hour = new Date().getHours();
    //     if (hour < 12) return "Good Morning";
    //     if (hour < 18) return "Good Afternoon";
    //     return "Good Evening";
    // }

    // fetch(`http://localhost:5000/getUsername?username=${localStorage.getItem("username")}`)
    // .then(response => response.json())
    // .then(data => {
    //     if (data.username) {
    //         document.getElementById("greeting").textContent = `${getGreeting()}, ${data.username}!`;
    //     } else {
    //         document.getElementById("greeting").textContent = `${getGreeting()}, Guest!`;
    //     }
    // })
    // .catch(error => console.error("Error fetching username:", error));

// Dynamic Greeting
function getGreeting() {
    const hour = new Date().getHours();
    if (hour < 12) return "Good Morning";
    if (hour < 18) return "Good Afternoon";
    return "Good Evening";
}

const username = localStorage.getItem("username");
if (username) {
    document.getElementById("greeting").textContent = `${getGreeting()}, ${username}!`;
} else {
    document.getElementById("greeting").textContent = `${getGreeting()}, Guest!`;
}


    // Weather fetch
    const API_KEY = "0b277494b999a70c1435c3da8ded982e"; // Replace with your OpenWeatherMap API Key
    const city = "kanchipuram"; // Change city based on user location

    async function fetchWeather() {
        try {
            const response = await fetch(
                `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${API_KEY}&units=metric`
            );
            const data = await response.json();

            if (data.cod === 200) {
                document.getElementById("weather").innerHTML = `
                    <p>${data.name}, ${data.sys.country}</p>
                    <p>${data.weather[0].description.toUpperCase()}</p>
                    <p>🌡️ ${data.main.temp}°C</p>
                    <p>💨 ${data.wind.speed} m/s</p>
                `;
            } else {
                document.getElementById("weather").innerHTML = `<p>Weather data unavailable</p>`;
            }
        } catch (error) {
            console.error("Weather API Error:", error);
        }
    }

    // Call the function when the page loads
    fetchWeather();

    // Check if health chart exists in HTML before initializing
    const chartElement = document.getElementById("healthChart");
    if (chartElement) {
        const ctx = chartElement.getContext("2d");
        const healthChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                datasets: [{
                    label: "Weight (kg)",
                    data: [72, 71.5, 71, 70.8, 70.5],
                    borderColor: "#16a085",
                    fill: false
                }]
            }
        });
    }

    // Disease information database as a fallback if API fails
    const diseaseInfo = {
        "Acne": {
            symptoms: "Whiteheads, blackheads, pimples, cysts, and nodules, primarily on the face, chest, and back.",
            causes: "Excess oil production, clogged hair follicles, bacteria, inflammation, and hormonal changes.",
            treatment: "Topical treatments (benzoyl peroxide, retinoids), oral antibiotics, isotretinoin for severe cases.",
            precautions: "Wash face twice daily, avoid touching face, use oil-free cosmetics, and maintain a healthy diet.",
            food_items: "Omega-3 rich foods, fruits and vegetables high in antioxidants, zinc-rich foods. Avoid dairy and high-glycemic foods."
        },
        "Eczema": {
            symptoms: "Dry, itchy, red, and inflamed skin with rough, scaly, or leathery patches that may ooze or crust over.",
            causes: "Genetic factors, immune system dysfunction, environmental triggers, and skin barrier defects.",
            treatment: "Moisturizers, topical corticosteroids, antihistamines, phototherapy, and immunosuppressants for severe cases.",
            precautions: "Use mild soaps, moisturize regularly, wear soft fabrics, avoid known triggers, and maintain optimal humidity levels.",
            food_items: "Anti-inflammatory foods like fatty fish, probiotics, flavonoid-rich fruits. Avoid common allergens like dairy, eggs, and nuts if sensitive."
        },
        "Psoriasis": {
            symptoms: "Red patches of skin covered with thick, silvery scales, dry/cracked skin that may bleed, itching, burning, or soreness.",
            causes: "Immune system dysfunction, genetic factors, environmental triggers, and stress.",
            treatment: "Topical treatments, phototherapy, oral or injected medications, and biologic drugs for severe cases.",
            precautions: "Moisturize regularly, avoid skin injuries, manage stress, limit alcohol, quit smoking, and protect skin from harsh weather.",
            food_items: "Anti-inflammatory foods like fatty fish, fruits, vegetables, whole grains. Limit red meat, dairy, and processed foods."
        },
        "Dermatitis": {
            symptoms: "Red, itchy rash, swelling, blisters that may ooze and crust over, and flaking skin.",
            causes: "Contact with allergens or irritants, genetic factors, environmental conditions, and stress.",
            treatment: "Avoidance of triggers, antihistamines, topical corticosteroids, and moisturizers.",
            precautions: "Identify and avoid triggers, use mild soaps, wear protective gloves when handling potential irritants, and keep skin moisturized.",
            food_items: "Anti-inflammatory foods, quercetin-rich foods like apples and onions. Avoid common food allergens if they trigger flare-ups."
        },
        "Melanoma": {
            symptoms: "Asymmetrical moles, irregular borders, varied colors, diameter larger than 6mm, and evolving appearance.",
            causes: "UV radiation exposure, genetic factors, fair skin, history of sunburns, and presence of numerous moles.",
            treatment: "Surgical removal, immunotherapy, targeted therapy, radiation therapy, and chemotherapy for advanced cases.",
            precautions: "Apply sunscreen regularly, avoid peak sun hours, wear protective clothing, perform regular skin checks, and get professional skin exams.",
            food_items: "Antioxidant-rich foods like berries, leafy greens, and foods high in selenium and vitamin D. Avoid processed foods and excessive alcohol."
        }
    };

    // Function to display disease information
    function displayDiseaseInfo() {
        const selectedDisease = document.getElementById("diseaseDropdown").value;
        if (!selectedDisease) {
            document.getElementById("diseaseDetails").classList.add("hidden");
            return;
        }

        // Try to fetch from API first
        fetch(`http://localhost:5000/get_disease_details?disease=${selectedDisease}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                updateDiseaseUI(selectedDisease, data);
            })
            .catch(error => {
                console.error("Error fetching disease details:", error);
                // Fallback to local data if API fails
                if (diseaseInfo[selectedDisease]) {
                    updateDiseaseUI(selectedDisease, diseaseInfo[selectedDisease]);
                } else {
                    document.getElementById("diseaseDetails").classList.add("hidden");
                    alert("Disease information not available. Please try another selection.");
                }
            });

        document.getElementById("diseaseDetails").classList.remove("hidden");
    }

    // Helper function to update the UI with disease details
    function updateDiseaseUI(diseaseName, data) {
        document.getElementById("diseaseName").innerText = diseaseName;
        document.getElementById("symptoms").innerText = data.symptoms || "No data available";
        document.getElementById("causes").innerText = data.causes || "No data available";
        document.getElementById("treatment").innerText = data.treatment || "No data available";
        document.getElementById("precautions").innerText = data.precautions || "No data available";
        document.getElementById("foodItems").innerText = data.food_items || "No data available";
        
        document.getElementById("diseaseDetails").classList.remove("hidden");
    }

    // Attach event listener to disease dropdown
    const diseaseDropdown = document.getElementById("diseaseDropdown");
    if (diseaseDropdown) {
        diseaseDropdown.addEventListener("change", displayDiseaseInfo);
    }
});