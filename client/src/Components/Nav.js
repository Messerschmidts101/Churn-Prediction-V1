import React from 'react'
import NavItem from './NavItem'
import { Link } from 'react-router-dom'
import { ReactComponent as Icon} from '../logo.svg'

function Nav({theme, }) {
    return (
        <nav className={'navbar navbar-expand-lg ' + theme}>
            <div className='container-fluid'>
                <Link to="/home" className="nav-heading navbar-brand">
                    <Icon style={{ heigh: "48px", width: "48px", margin: "0 5px" }} /><wbr />
                    TelCo <wbr />Generic
                </Link>
                <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                    <span className="navbar-toggler-icon"></span>
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

export default Nav