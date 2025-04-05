const form=document.getElementById("form")
const passError=document.getElementById("pass-error")
const emailError=document.getElementById("email-error")
const email = document.getElementById('email')
const password = document.getElementById('pass')
function validateForm() {
    if (email.value.trim() === '') {
        emailError.style.display="flex"
        emailError.innerHTML='email is required'
        return false;
    }

    // Validate password (required and minimum length of 6 characters)
    if (password.value.trim() === '') {
        emailError.style.display="flex"
        emailError.innerHTML='Password is required' 
        return false; 
    } else if (password.value.length <= 6) {
        emailError.style.display="flex"
        emailError.innerHTML='Password must be at least 6 characters';
        return false; // Prevent form submission
    }
    return true;
}

function reset(){
    emailError.style.display="none"
    emailError.innerHTML=''
    passError.style.display="none"
    passError.innerHTML=''
}

async function fetchData(userData){
    const data=await fetch("http://127.0.0.1:5000/login",{
        method:"POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body:JSON.stringify(userData)
    })
    if(!data.ok) throw new Error("error while fetching")
        console.log(userData)
    return await data.json()
}


form.addEventListener('submit',async (e)=>{
    e.preventDefault();
    const success_msg=document.getElementById("success-msg");
    if(validateForm()){
        reset();
        let {message}=await fetchData({email:email.value , password:password.value})
        success_msg.style.display='flex'
        success_msg.innerHTML=`${message}`
        setTimeout(() => {
            window.location.href = '/pages/LoginPage.html';
        }, 2000); 
    }
    
});