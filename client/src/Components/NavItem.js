import React from 'react'
import { Link } from 'react-router-dom'

function NavItem({children, toLink}) {
    return (
        <li className='nav-item active'>
            <Link to={toLink} className="nav-link">
            {children}
            </Link>
        </li>
    )
}

export default NavItem