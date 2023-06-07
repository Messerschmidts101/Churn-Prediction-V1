import React from 'react'

function Header1({children, className}) {
    return (
        <h1 className={className}>{children}</h1>
    )
}

export default Header1