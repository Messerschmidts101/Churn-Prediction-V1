import React from 'react'
import NavItem from './NavItem'
import { Link } from 'react-router-dom'

function Nav({theme, }) {
    return (
        <nav className={'navbar navbar-expand-lg ' + theme}>
            <div className='container-fluid'>
                <Link to="/home" className="nav-heading navbar-brand">The Churners</Link>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div id='navbarSupportedContent' className='collapse navbar-collapse'>
                    <ul className='navbar-nav mr-auto nav-item-custom'>
                        <NavItem toLink="/single_customer">Single Customer</NavItem>
                        <NavItem toLink="/multiple_customer">Customer Dataset</NavItem>
                    </ul>
                </div>
            </div>
        </nav>
    )
}
//<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
{/* <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbar">
<span class="navbar-toggler-icon"></span>
</button>
<div class="collapse navbar-collapse" id="navbar">
<div class="navbar-nav">
    <a class="nav-item nav-link" id="home" href="/">Home</a>
    {% if current_user.is_authenticated %}
        <a class="nav-item nav-link" id="logout" href="{{ url_for('auth.logout') }}">Logout</a>
    {% else %}
        <a class="nav-item nav-link" id="login" href="/login">Login</a>
        <a class="nav-item nav-link" id="signUp" href="/signUp">Sign Up</a>
    {% endif %}
</div>
</div>
</nav> */}
export default Nav