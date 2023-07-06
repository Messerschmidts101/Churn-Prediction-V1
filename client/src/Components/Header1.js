import React from 'react'

function Header1({children, className, id}) {
    return (
        <h1 id={id} className={className}>{children}</h1>
    )
}

export default Header1