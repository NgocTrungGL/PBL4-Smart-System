const logregBox = document.querySelector('.logreg-box');  // Chú ý dấu chấm '.' trước lớp CSS
const loginLink = document.querySelector('.login-link');  // Thêm dấu chấm '.' trước lớp CSS
const registerLink = document.querySelector('.register-link');  // Thêm dấu chấm '.' trước lớp CSS

registerLink.addEventListener('click', () => {
    logregBox.classList.add('active');
});

loginLink.addEventListener('click', () => {
    logregBox.classList.remove('active');
});

