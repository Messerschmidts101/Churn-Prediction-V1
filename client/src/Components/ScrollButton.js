import React from 'react'

function ScrollButton({children, theme, href, className}) {
    return (
        <a className={'btn btn-' + theme + " " + className} href={href} role={"button"}>{children}</a>
    )
}

export default ScrollButton