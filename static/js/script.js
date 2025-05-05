// Initialize EmailJS
emailjs.init("94B2WYPA30VrIP_OM"); // Public Key

// Toggle between Login and Register forms
function toggleForm(form) {
  document.getElementById('login-form').style.display = form === 'login' ? 'flex' : 'none';
  document.getElementById('register-form').style.display = form === 'register' ? 'flex' : 'none';
}


// Handle Register Form Submission
document.getElementById('registerForm').addEventListener('submit', function (event) {
  event.preventDefault();

  const fullName = document.getElementById('fullName').value;
  const preferredName = document.getElementById('preferredName').value;
  const dob = document.getElementById('dob').value;
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;

  // Generate Username based on DOB and Preferred Name
  const dobParts = dob.split("-");
  const username = `${preferredName}@${dobParts[2]}${dobParts[1]}`;  // username as preferredName@dob

  // Prepare email parameters for sending the registration details
  const emailParams = {
    full_name: fullName,
    username: username,
    email: email,
    password: password
  };

  // Send email with username and password
  emailjs
    .send("service_c4nbbdh", "template_0mgr93k", emailParams)
    .then(
      (response) => {
        console.log("Email sent successfully:", response);
        alert(`Your username and details have been emailed to: ${email}`);
      },
      (error) => {
        console.error("Email failed to send:", error);
        alert("Failed to send email. Please try again.");
      }
    );

  // Prepare user data to be sent to backend
  const userData = { fullName, preferredName, dob, email, password, username };

  // Send user data to the backend to save in MongoDB
fetch('http://localhost:5000/register', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    },
    body: JSON.stringify({
        fullName: fullName,
        preferredName: preferredName,
        dob: dob,
        email: email,
        password: password
    }),
    credentials: 'include'  // 🔹 Allow credentials for CORS
})

    .then((response) => response.json())
    .then((data) => {
      if (data.message === 'User registered successfully') {
        alert('Registration successful!');
        document.getElementById('registerForm').reset();
        toggleForm('login');  // Switch to login form after successful registration
      } else {
        alert('Failed to register user. Please try again.');
      }
    })
    .catch((error) => {
      console.error('Error:', error);
      alert('An error occurred. Please try again.');
    });
});

// Handle Login Form Submission
document.getElementById('loginForm').addEventListener('submit', function (event) {
  event.preventDefault();
  
  const email = document.getElementById('loginEmail').value; // Use email for login
  const password = document.getElementById('loginPassword').value;

  // Send login credentials to backend for validation
  if (email && password) {
    fetch('http://localhost:5000/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),  // Send email and password for login
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.message === 'Login successful') {
          alert('Login successful!');
          // You can redirect to another page or show a success message here
        } else {
          alert(data.message);  // Display error message from backend
        }
      })
      .catch((error) => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
      });
  } else {
    alert("Please enter both email and password.");
  }
});

// Js Function for Validating Preferred Name (kept as is)
let errorTimer;
function validatePreferredName(input) {
  const regex = /^[A-Za-z]+$/;
  const errorElement = document.getElementById('errorPreferredName');
  if (!regex.test(input.value)) {
    errorElement.style.display = 'block';
    input.value = input.value.replace(/[^A-Za-z]/g, '');
    if (errorTimer) clearTimeout(errorTimer);
    errorTimer = setTimeout(() => {
      errorElement.style.display = 'none';
    }, 5000);
  } else {
    errorElement.style.display = 'none';
  }
}

document.addEventListener('DOMContentLoaded', function () {
    // Login form event listener
    document.getElementById('loginForm').addEventListener('submit', async function (event) {
        event.preventDefault();

        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;

        try {
            const response = await fetch('http://localhost:5000/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();

            if (response.ok) {
                localStorage.setItem('username', data.username);  // Store username
                window.location.href = "index.html";  // Redirect to index page
            } else {
                alert(data.message);  // Show error message
            }
        } catch (error) {
            console.error('Login error:', error);
            alert('Server error. Please try again.');
        }
    });
});


// Validate username (only alphabets allowed)
function validatePreferredName(input) {
    const errorElement = document.getElementById('errorPreferredName');
    if (!/^[A-Za-z]+$/.test(input.value)) {
        errorElement.style.display = 'block';
    } else {
        errorElement.style.display = 'none';
    }
}



document.getElementById('loginForm').addEventListener('submit', async function (e) {
    e.preventDefault(); // prevent form reload

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();
        if (response.ok) {
            console.log('Login success:', data);

            localStorage.setItem('username', data.username); // save username
            window.location.href = '/'; // redirect to index
        } else {
            alert(data.message || 'Login failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Server error. Please try again later.');
    }
});

