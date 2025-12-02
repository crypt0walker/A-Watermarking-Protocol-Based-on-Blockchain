// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract WatermarkProtocol {
    struct TxTokens {
        bytes Xd;
        uint256 TX;
        bytes pkX_RA;
        bytes encN;
        bytes sigRA;
        bytes sigCP;
        address buyer;
        bool used;
    }

    mapping(bytes32 => TxTokens) public txByRaToken;

    event TokenRegistered(bytes32 indexed raTokenId, address indexed cp);
    event TokenConfirmed(bytes32 indexed raTokenId, address indexed buyer);

    function registerByCP(
        bytes calldata Xd,
        uint256 TX_,
        bytes calldata pkX_RA,
        bytes calldata encN,
        bytes calldata sigRA,
        bytes calldata sigCP,
        address buyer
    ) external {
        bytes32 tokenId = keccak256(abi.encode(pkX_RA, encN));
        require(txByRaToken[tokenId].buyer == address(0), "exists");
        txByRaToken[tokenId] = TxTokens(Xd, TX_, pkX_RA, encN, sigRA, sigCP, buyer, false);
        emit TokenRegistered(tokenId, msg.sender);
    }

    function confirmByB(
        bytes32 tokenId,
        bytes calldata Xd,
        uint256 TX_,
        bytes calldata pkX_RA,
        bytes calldata encN,
        bytes calldata sigRA,
        bytes calldata sigCP
    ) external {
        TxTokens storage t = txByRaToken[tokenId];
        require(!t.used, "used");
        require(t.buyer == msg.sender, "not buyer");
        require(
            keccak256(t.Xd) == keccak256(Xd) &&
                t.TX == TX_ &&
                keccak256(t.pkX_RA) == keccak256(pkX_RA) &&
                keccak256(t.encN) == keccak256(encN),
            "mismatch"
        );
        t.used = true;
        emit TokenConfirmed(tokenId, msg.sender);
    }
}
