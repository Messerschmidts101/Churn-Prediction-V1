import React from 'react'

function Button({children, theme, href, className, onClick}) {
    return (
        <button className={'btn btn-' + theme + " " + className} href={href} onClick={onClick}>{children}</button>
    )
}

export default Button