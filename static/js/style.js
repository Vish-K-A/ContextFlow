const sidebar = document.getElementById('sidebar');

function toggleSidebar() {
  sidebar.classList.toggle('close');
}

function toggleSubMenu(button) {
  const subMenu = button.nextElementSibling;
  subMenu.classList.toggle('show');
  
  button.classList.toggle('rotate');
}